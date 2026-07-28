import json
from datetime import datetime
from types import SimpleNamespace

import pytest
from conductor_core import GenerationRequest
from mido import Message, MidiFile

import conductor_eval.evaluator as evaluator_module
from conductor_eval import EvalEngineAdapter, Evaluator


def test_texture_checks_are_available():
    assert {"monophony", "polyphony"} <= Evaluator.AVAILABLE_TESTS.keys()


def test_harmonic_checks_are_available():
    assert {
        "chord_progression",
        "harmonic_rhythm",
        "chord_event_positions",
    } <= Evaluator.AVAILABLE_TESTS.keys()


class RecordingEngine:
    def __init__(self, result):
        self.result = result
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return self.result


def test_eval_engine_adapter_delegates_generation_to_core(tmp_path):
    midi_path = tmp_path / "core-loop.mid"
    MidiFile().save(midi_path)
    core_result = SimpleNamespace(
        midi_path=str(midi_path),
        messages=[{"role": "assistant", "content": "loop"}],
        cost=0.125,
    )
    engine = RecordingEngine(core_result)
    adapter = EvalEngineAdapter(tmp_path / "core-artifacts", engine=engine)

    midi, messages, cost = adapter.generate(
        description="warm quarter-note arpeggio",
        key="C",
        scale="major",
        model="gpt-test",
        temperature=0.2,
        use_thinking=True,
        effort="medium",
    )

    assert isinstance(midi, MidiFile)
    assert midi_path.exists()
    assert messages == core_result.messages
    assert cost == core_result.cost
    assert engine.requests == [
        GenerationRequest(
            key="C",
            scale="major",
            description="warm quarter-note arpeggio",
            model="gpt-test",
            temperature=0.2,
            use_thinking=True,
            effort="medium",
            render_audio=False,
        )
    ]


def test_save_results_uses_unique_safe_task_directory(tmp_path):
    evaluator = Evaluator(output_dir=str(tmp_path / "evaluations"))
    run_path = tmp_path / "run"
    midi = MidiFile()
    messages = [{"role": "user", "content": "prompt"}]
    result = {
        "model": "gpt-test",
        "provider": "OpenAI",
        "tests": {"overall_pass": True},
    }
    task = {
        "provider": "OpenAI",
        "model": "gpt-test",
        "original_prompt": "warm loop",
        "root": "C",
        "scale": "major",
        "variation_name": "standard",
        "task_id": "task-warm_loop-0123456789abcdef-1",
    }

    evaluator._save_results(result, midi, messages, run_path, task)

    result_dir = run_path / "results" / task["task_id"]
    assert (result_dir / "loop.mid").exists()
    legacy_filename = "output" + ".mid"
    assert not (result_dir / legacy_filename).exists()
    assert json.loads((result_dir / "messages.json").read_text(encoding="utf-8")) == messages


def test_create_run_directory_returns_compact_authoritative_metadata(tmp_path, monkeypatch):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    frozen_time = datetime(2026, 7, 28, 12, 34, 56, 789012)
    monkeypatch.setattr(evaluator_module, "datetime", SimpleNamespace(now=lambda: frozen_time))
    monkeypatch.setattr(
        evaluator_module,
        "uuid4",
        lambda: SimpleNamespace(hex="0123456789abcdef" * 2),
    )

    run_path, run_id, timestamp = evaluator._create_run_directory("same name")

    assert timestamp == "20260728_123456_789012"
    assert run_id == "20260728_123456_789012_same_name-f03c3f373761_0123456789abcdef"
    assert run_path.name == run_id
    assert run_path.is_dir()


def test_create_run_directory_fails_for_exact_collision(tmp_path, monkeypatch):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    monkeypatch.setattr(
        evaluator_module,
        "datetime",
        SimpleNamespace(now=lambda: datetime(2026, 7, 28, 12, 34, 56, 789012)),
    )
    monkeypatch.setattr(
        evaluator_module,
        "uuid4",
        lambda: SimpleNamespace(hex="0123456789abcdef" * 2),
    )

    evaluator._create_run_directory("same name")

    with pytest.raises(FileExistsError):
        evaluator._create_run_directory("same name")


def test_generate_tasks_uses_distinct_ids_for_normalization_collisions(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    tasks = evaluator._generate_tasks(
        prompts=["warm/loop", "warm_loop"],
        roots=["C"],
        resolved_models=[("OpenAI", "gpt-test")],
        tests=["scale"],
        test_reasoning=False,
    )

    assert len({task["task_id"] for task in tasks}) == len(tasks)
    assert all(task["task_id"].startswith("task-warm_loop-") for task in tasks)
    assert all(len(task["task_id"].split("-")[-2]) == 16 for task in tasks)


def test_generate_tasks_qualifies_repeated_identical_tasks_with_occurrences(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    tasks = evaluator._generate_tasks(
        prompts=["same prompt"] * 3,
        roots=["C"],
        resolved_models=[("OpenAI", "gpt-test")],
        tests=["scale"],
        test_reasoning=False,
    )

    major_ids = [task["task_id"] for task in tasks if task["scale"] == "major"]
    assert [task_id.rsplit("-", 1)[-1] for task_id in major_ids] == ["1", "2", "3"]


def test_save_results_uses_compact_task_directory_and_fails_on_collision(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    run_path = tmp_path / "run"
    task = {
        "provider": "OpenAI",
        "model": "org/model:v1",
        "original_prompt": "x" * 500,
        "root": "C",
        "scale": "major",
        "variation_name": "standard",
        "task_id": "task-" + ("x" * 18) + "-0123456789abcdef-1",
    }

    evaluator._save_results({"task_id": task["task_id"]}, None, [], run_path, task)

    result_dir = run_path / "results" / task["task_id"]
    assert (result_dir / "test_results.json").exists()
    assert len(str(result_dir.relative_to(run_path))) < 100

    with pytest.raises(FileExistsError):
        evaluator._save_results({"task_id": task["task_id"]}, None, [], run_path, task)


def test_run_tests_routes_polyphony_params_and_updates_overall_pass(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    midi = MidiFile(ticks_per_beat=480)
    track = midi.add_track()
    track.append(Message("note_on", note=60, velocity=80, time=0))
    track.append(Message("note_on", note=67, velocity=80, time=0))
    track.append(Message("note_off", note=60, velocity=0, time=480))
    track.append(Message("note_off", note=67, velocity=0, time=0))

    results = evaluator.run_tests(
        midi_data=midi,
        root="C",
        scale="major",
        prompt="block chords",
        tests=["polyphony"],
        test_params={"polyphony": {"min_voices": 3}},
    )

    assert results["polyphony"]["params"] == {"min_voices": 3}
    assert results["polyphony"]["passed"] is False
    assert results["overall_pass"] is False


def test_run_tests_injects_root_and_scale_into_chord_progression(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    midi = MidiFile(ticks_per_beat=480)
    track = midi.add_track()
    for pitch in [60, 64, 67]:
        track.append(Message("note_on", note=pitch, velocity=80, time=0))
    track.append(Message("note_off", note=60, velocity=0, time=480))
    track.append(Message("note_off", note=64, velocity=0, time=0))
    track.append(Message("note_off", note=67, velocity=0, time=0))

    results = evaluator.run_tests(
        midi_data=midi,
        root="C",
        scale="major",
        prompt="one C major chord",
        tests=["chord_progression"],
        test_params={
            "chord_progression": {
                "progression": ["I"],
                "beats_per_chord": 1,
            }
        },
    )

    assert results["chord_progression"]["passed"] is True
    assert results["chord_progression"]["params"] == {
        "progression": ["I"],
        "beats_per_chord": 1,
        "root": "C",
        "scale": "major",
    }
    assert results["overall_pass"] is True


def test_run_tests_uses_explicit_duration_before_prompt_detection(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    midi = MidiFile(ticks_per_beat=480)
    track = midi.add_track()
    track.append(Message("note_on", note=60, velocity=80, time=0))
    track.append(Message("note_off", note=60, velocity=0, time=480))

    results = evaluator.run_tests(
        midi_data=midi,
        root="C",
        scale="major",
        prompt="use eighth notes",
        tests=["duration"],
        test_params={"duration": {"duration": "quarter"}},
    )

    assert results["duration"]["incorrect"] == 0
    assert results["duration"]["params"] == {"duration": "quarter"}
    assert results["duration"]["detected_from_prompt"] is False


def test_run_tests_keeps_duration_prompt_detection_as_fallback(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    midi = MidiFile(ticks_per_beat=480)
    track = midi.add_track()
    track.append(Message("note_on", note=60, velocity=80, time=0))
    track.append(Message("note_off", note=60, velocity=0, time=240))

    results = evaluator.run_tests(
        midi_data=midi,
        root="C",
        scale="major",
        prompt="use eighth notes",
        tests=["duration"],
    )

    assert results["duration"]["incorrect"] == 0
    assert results["duration"]["params"] == {"duration": "eighth"}
    assert results["duration"]["detected_from_prompt"] is True


def test_generate_tasks_copies_test_params_to_each_task(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    test_params = {"polyphony": {"min_voices": 3}}

    tasks = evaluator._generate_tasks(
        prompts=["block chords"],
        roots=["C"],
        resolved_models=[("Ollama", "test-model")],
        tests=["polyphony"],
        test_reasoning=False,
        test_params=test_params,
    )

    assert len(tasks) == 2
    assert all(task["test_params"] == test_params for task in tasks)


def test_test_params_reject_unselected_test(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")

    with pytest.raises(ValueError, match="unselected tests: polyphony"):
        evaluator.run_tests(
            midi_data=MidiFile(),
            root="C",
            scale="major",
            prompt="melody",
            tests=["scale"],
            test_params={"polyphony": {"min_voices": 3}},
        )


def test_summary_preserves_unknown_costs_and_excludes_failed_latency(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    summary = evaluator._generate_summary(
        [
            {
                "model": "reported",
                "provider": "OpenAI",
                "root": "C",
                "scale": "major",
                "metrics": {"cost": 0.25, "api_latency": 2.0, "attempt_latency": 2.0},
                "tests": {"overall_pass": True},
                "error": None,
            },
            {
                "model": "unknown",
                "provider": "Ollama",
                "root": "C",
                "scale": "major",
                "metrics": {"cost": None, "api_latency": None, "attempt_latency": 7.0},
                "tests": {"overall_pass": False},
                "error": "timed out",
            },
        ],
        {"timestamp": "20260723_000000", "run_name": "metric-contract"},
    )

    assert summary["totals"]["total_cost"] == 0.25
    assert summary["totals"]["known_cost_generations"] == 1
    assert summary["totals"]["unknown_cost_generations"] == 1
    assert summary["totals"]["total_time"] == 9.0
    assert summary["totals"]["avg_successful_latency"] == 2.0
    assert summary["by_model"]["unknown"]["avg_latency"] is None


def test_failed_generation_records_attempt_latency(monkeypatch, tmp_path):
    class FailingAdapter:
        def __init__(self, output_dir):
            self.output_dir = output_dir

        def generate(self, **kwargs):
            raise RuntimeError("provider timed out")

    monkeypatch.setattr("conductor_eval.evaluator.EvalEngineAdapter", FailingAdapter)
    monkeypatch.setattr("conductor_eval.evaluator.time.perf_counter", lambda: next(clock))
    clock = iter([100.0, 103.5])
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    task = {
        "provider": "OpenAI",
        "model": "test-model",
        "full_prompt": "prompt",
        "original_prompt": "prompt",
        "root": "C",
        "scale": "major",
        "use_thinking": False,
        "effort": None,
        "variation_name": "standard",
        "task_id": "task-prompt-0123456789abcdef-1",
    }

    result = evaluator._run_single(task, tmp_path / "run", ["scale"])

    assert result["error"] == "provider timed out"
    assert result["metrics"] == {
        "api_latency": None,
        "attempt_latency": 3.5,
        "cost": None,
        "cost_available": False,
    }

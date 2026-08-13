import json
import logging
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import pytest
from conductor_core import GenerationRequest
from conductor_core.errors import ProviderRateLimitError
from mido import Message, MidiFile

import conductor_eval.evaluator as evaluator_module
from conductor_eval import EvalEngineAdapter, Evaluator
from conductor_eval.outcomes import get_overall_status


def test_texture_checks_are_available():
    assert {"monophony", "polyphony"} <= Evaluator.AVAILABLE_TESTS.keys()


def test_harmonic_checks_are_available():
    assert {
        "chord_progression",
        "harmonic_rhythm",
        "chord_event_positions",
    } <= Evaluator.AVAILABLE_TESTS.keys()


def test_ollama_discovery_returns_models(monkeypatch):
    monkeypatch.setattr(
        "conductor_eval.evaluator.ollama_api.get_model_list",
        lambda: ["llama3.2", "qwen3"],
    )

    assert Evaluator._discover_ollama_models() == ["llama3.2", "qwen3"]


def test_ollama_discovery_returns_empty_when_unavailable(monkeypatch):
    def fail_discovery():
        raise RuntimeError("Ollama is unavailable")

    monkeypatch.setattr("conductor_eval.evaluator.ollama_api.get_model_list", fail_discovery)

    assert Evaluator._discover_ollama_models() == []


def test_evaluator_initialization_preserves_root_logging(tmp_path):
    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_handlers = tuple(root_logger.handlers)
    sentinel = logging.NullHandler()
    root_logger.addHandler(sentinel)

    try:
        output_dir = tmp_path / "evaluations"
        Evaluator(output_dir=output_dir)

        assert root_logger.level == original_level
        assert tuple(root_logger.handlers) == (*original_handlers, sentinel)
        assert not output_dir.exists()
    finally:
        root_logger.removeHandler(sentinel)


def test_successful_evaluation_log_is_minimal(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")

    evaluator.evaluate(
        prompts="warm loop",
        roots=["C"],
        models=[],
        run_name="minimal",
    )

    run_path = next((tmp_path / "evaluations").iterdir())
    log_contents = (run_path / "run.log").read_text(encoding="utf-8")

    assert "Starting evaluation 'minimal' with 0 total tasks" in log_contents
    assert "Evaluation complete. Results saved to" in log_contents
    assert "Generation completed" not in log_contents
    assert "Checks completed" not in log_contents
    assert "Saved result artifacts" not in log_contents


def test_evaluations_keep_logs_and_artifacts_isolated_when_overlapping(monkeypatch, tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    resolution_barrier = threading.Barrier(2)
    task_barrier = threading.Barrier(2)
    errors = []
    run_uuids = iter(["a" * 32, "b" * 32])

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 26, 12, 34, 56, 789012, tzinfo=tz)

    def resolve_models(models, logger):
        resolution_barrier.wait(timeout=5)
        marker = threading.current_thread().name
        logger.info("Resolving models for %s", marker)
        return [("Ollama", marker)]

    def generate_tasks(**kwargs):
        marker = kwargs["resolved_models"][0][1]
        return [{"provider": "Ollama", "model": marker, "marker": marker}]

    def run_single(task, run_path, tests_to_run, logger):
        task_barrier.wait(timeout=5)
        logger.info("Running task marker=%s", task["marker"])
        return {
            "provider": task["provider"],
            "model": task["model"],
            "root": "C",
            "scale": "major",
            "metrics": {
                "api_latency": 1.0,
                "attempt_latency": 1.0,
                "cost": 0.0,
            },
            "tests": {"overall_pass": True},
            "error": None,
        }

    monkeypatch.setattr("conductor_eval.evaluator.datetime", FixedDatetime)
    monkeypatch.setattr(
        "conductor_eval.evaluator.uuid4",
        lambda: SimpleNamespace(hex=next(run_uuids)),
    )
    monkeypatch.setattr(evaluator, "_resolve_models", resolve_models)
    monkeypatch.setattr(evaluator, "_generate_tasks", generate_tasks)
    monkeypatch.setattr(evaluator, "_run_single", run_single)

    def run_evaluation():
        try:
            evaluator.evaluate(
                prompts="warm loop",
                roots=["C"],
                models="none",
                run_name="same-name",
            )
        except Exception as error:
            errors.append(error)

    threads = [
        threading.Thread(
            target=run_evaluation,
            name=f"evaluation-{index}",
        )
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert not errors

    run_paths = list((tmp_path / "evaluations").iterdir())
    assert len(run_paths) == 2
    assert len({path.name for path in run_paths}) == 2
    assert all(path.name.startswith("20260726_123456_789012_same-name-") for path in run_paths)

    for run_path in run_paths:
        log_contents = (run_path / "run.log").read_text(encoding="utf-8")
        config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
        other_paths = [path for path in run_paths if path != run_path]
        marker = config["models"][0][1]
        other_marker = next(
            json.loads((path / "config.json").read_text(encoding="utf-8"))["models"][0][1]
            for path in other_paths
        )
        assert config["run_id"] == run_path.name
        assert summary["run_id"] == run_path.name
        assert "Starting evaluation 'same-name'" in log_contents
        assert f"Running task marker={marker}" in log_contents
        assert f"Running task marker={other_marker}" not in log_contents
        assert str(run_path) in log_contents
        assert all(str(other_path) not in log_contents for other_path in other_paths)


def test_failed_evaluation_retains_and_closes_its_run_log(monkeypatch, tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    created_logger = None
    created_handler = None
    create_run_logger = evaluator._create_run_logger

    def fail_resolution(models, logger):
        raise RuntimeError("model resolution failed")

    def track_run_logger(run_path, run_id):
        nonlocal created_logger, created_handler
        created_logger, created_handler = create_run_logger(run_path, run_id)
        return created_logger, created_handler

    monkeypatch.setattr(evaluator, "_resolve_models", fail_resolution)
    monkeypatch.setattr(evaluator, "_create_run_logger", track_run_logger)

    with pytest.raises(RuntimeError, match="model resolution failed"):
        evaluator.evaluate(
            prompts="warm loop",
            roots=["C"],
            models="none",
            run_name="failure",
        )

    run_path = next((tmp_path / "evaluations").iterdir())
    log_contents = (run_path / "run.log").read_text(encoding="utf-8")

    assert "Evaluation failed: run_path=" in log_contents
    assert "error_type=RuntimeError" in log_contents
    assert "Traceback:" in log_contents
    assert "model resolution failed" not in log_contents
    assert created_logger is not None
    assert created_handler is not None
    assert created_logger.name not in logging.Logger.manager.loggerDict
    assert not created_logger.handlers
    assert created_handler.stream is None


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


def test_run_tests_marks_empty_midi_ineligible_under_default_checks(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")

    results = evaluator.run_tests(
        midi_data=MidiFile(ticks_per_beat=480),
        root="C",
        scale="major",
        prompt="use quarter notes",
        tests=["scale", "duration"],
    )

    assert results["scale"]["total"] == 0
    assert results["scale"]["status"] == "ineligible"
    assert results["duration"]["total"] == 0
    assert results["duration"]["status"] == "ineligible"
    assert results["overall_pass"] is False
    assert results["overall_status"] == "ineligible"


def test_run_tests_marks_dangling_note_ineligible_under_default_checks(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    midi = MidiFile(ticks_per_beat=480)
    track = midi.add_track()
    track.append(Message("note_on", note=60, velocity=80, time=0))

    results = evaluator.run_tests(
        midi_data=midi,
        root="C",
        scale="major",
        prompt="use quarter notes",
        tests=["scale", "duration"],
    )

    assert results["scale"]["total"] == 0
    assert results["scale"]["status"] == "ineligible"
    assert results["duration"]["total"] == 0
    assert results["duration"]["status"] == "ineligible"
    assert results["overall_pass"] is False
    assert results["overall_status"] == "ineligible"


@pytest.mark.parametrize("texture_test", ["monophony", "polyphony"])
def test_run_tests_requires_completed_notes_for_texture_evidence(tmp_path, texture_test):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")

    results = evaluator.run_tests(
        midi_data=MidiFile(ticks_per_beat=480),
        root="C",
        scale="major",
        prompt="empty texture",
        tests=[texture_test],
    )

    assert results["scale"]["status"] == "ineligible"
    assert results[texture_test]["total_notes"] == 0
    assert results[texture_test]["eligible"] is False
    assert results[texture_test]["status"] == "ineligible"
    assert results["overall_pass"] is False
    assert results["overall_status"] == "ineligible"


def test_run_tests_marks_skipped_only_selection_ineligible(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")

    results = evaluator.run_tests(
        midi_data=MidiFile(ticks_per_beat=480),
        root="C",
        scale="major",
        prompt="melody",
        tests=["duration"],
    )

    assert results["scale"]["status"] == "ineligible"
    assert results["duration"]["status"] == "ineligible"
    assert results["overall_pass"] is False
    assert results["overall_status"] == "ineligible"


def test_run_tests_always_includes_scale_when_callers_omit_it(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    midi = MidiFile(ticks_per_beat=480)
    track = midi.add_track()
    track.append(Message("note_on", note=60, velocity=80, time=0))
    track.append(Message("note_off", note=60, velocity=0, time=480))

    results = evaluator.run_tests(
        midi_data=midi,
        root="C",
        scale="major",
        prompt="melody",
        tests=["duration"],
    )

    assert results["scale"]["passed"] is True
    assert results["duration"]["status"] == "ineligible"
    assert results["overall_pass"] is True
    assert results["overall_status"] == "passed"


def test_run_tests_classifies_checker_exceptions_as_ineligible_check_errors(monkeypatch, tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")

    def fail_scale_check(*args):
        raise RuntimeError("checker failed")

    monkeypatch.setattr(
        evaluator,
        "AVAILABLE_TESTS",
        {**evaluator.AVAILABLE_TESTS, "scale": fail_scale_check},
    )

    results = evaluator.run_tests(
        midi_data=MidiFile(ticks_per_beat=480),
        root="C",
        scale="major",
        prompt="melody",
        tests=["scale"],
    )

    assert results["scale"]["ran"] is False
    assert results["scale"]["eligible"] is False
    assert results["scale"]["status"] == "check_error"
    assert results["overall_pass"] is False
    assert results["overall_status"] == "check_error"


@pytest.mark.parametrize(
    ("test_results", "expected"),
    [
        ({"overall_pass": True, "overall_status": "passed"}, True),
        ({"overall_pass": False, "overall_status": "failed"}, True),
        ({"overall_pass": False, "overall_status": "ineligible"}, False),
        ({"overall_pass": False, "overall_status": "generation_error"}, False),
        ({"overall_pass": False, "overall_status": "rate_limited"}, False),
        ({"overall_pass": False, "overall_status": "check_error"}, False),
        ({"overall_pass": False}, True),
    ],
)
def test_overall_eligibility_contract_includes_only_valid_verdicts(test_results, expected):
    assert Evaluator._is_overall_eligible(test_results) is expected


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"tests": {"overall_status": "rate_limited"}, "error": "throttled"}, "rate_limited"),
        ({"tests": {}, "error": "provider failed"}, "generation_error"),
        ({"tests": {"overall_pass": True}, "error": None}, "passed"),
        ({"tests": {"overall_pass": False}, "error": None}, "failed"),
        ({}, "failed"),
    ],
)
def test_get_overall_status_preserves_status_and_supports_legacy_results(result, expected):
    assert get_overall_status(result) == expected


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "4"])
def test_evaluate_rejects_invalid_cloud_concurrency_before_output(tmp_path, value):
    output_dir = tmp_path / "evaluations"
    evaluator = Evaluator(output_dir=output_dir)

    with pytest.raises(ValueError, match="max_cloud_concurrency must be a positive integer"):
        evaluator.evaluate(
            prompts="melody",
            roots=["C"],
            models=[],
            run_name="invalid-concurrency",
            max_cloud_concurrency=value,
        )

    assert not output_dir.exists()


def test_async_batch_caps_each_provider_independently(monkeypatch, tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    active = {"OpenAI": 0, "Anthropic": 0}
    peaks = {"OpenAI": 0, "Anthropic": 0}
    total_active = 0
    total_peak = 0
    lock = threading.Lock()

    def run_single(task, **kwargs):
        nonlocal total_active, total_peak
        provider = task["provider"]
        with lock:
            active[provider] += 1
            total_active += 1
            peaks[provider] = max(peaks[provider], active[provider])
            total_peak = max(total_peak, total_active)
        time.sleep(0.05)
        with lock:
            active[provider] -= 1
            total_active -= 1
        return {
            "provider": provider,
            "model": task["model"],
            "root": "C",
            "scale": "major",
            "metrics": {},
            "tests": {"overall_pass": True, "overall_status": "passed"},
            "error": None,
        }

    monkeypatch.setattr(evaluator, "_run_single", run_single)
    tasks = [
        {"provider": provider, "model": f"model-{index}"}
        for provider in active
        for index in range(4)
    ]

    results = evaluator_module.asyncio.run(
        evaluator._run_async_batch(
            tasks,
            tmp_path,
            ["scale"],
            logging.Logger("test"),
            max_cloud_concurrency=2,
        )
    )

    assert len(results) == 8
    assert peaks == {"OpenAI": 2, "Anthropic": 2}
    assert total_peak == 4


def test_summary_separates_all_outcomes_and_excludes_noneligible_results(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    statuses = [
        "passed",
        "failed",
        "ineligible",
        "generation_error",
        "rate_limited",
        "check_error",
    ]
    results = [
        {
            "model": "model",
            "provider": "OpenAI",
            "root": "C",
            "scale": "major",
            "metrics": {},
            "tests": {
                "overall_pass": status == "passed",
                "overall_status": status,
            },
            "error": (
                "provider throttled"
                if status == "rate_limited"
                else "generation failed"
                if status == "generation_error"
                else None
            ),
        }
        for status in statuses
    ]

    summary = evaluator._generate_summary(results, {"run_id": "outcome-test"})

    assert summary["totals"]["eligible_generations"] == 2
    assert summary["totals"]["overall_pass_count"] == 1
    assert summary["totals"]["validation_failed_generations"] == 1
    assert summary["totals"]["ineligible_generations"] == 1
    assert summary["totals"]["generation_error_generations"] == 1
    assert summary["totals"]["rate_limited_generations"] == 1
    assert summary["totals"]["check_error_generations"] == 1
    assert summary["totals"]["overall_pass_rate"] == 0.5
    assert summary["by_model"]["model"]["check_errors"] == 1
    assert summary["by_model"]["model"]["rate_limited"] == 1
    assert summary["by_root"]["C"]["check_errors"] == 1
    assert summary["by_scale"]["major"]["check_errors"] == 1


def test_rate_limit_error_is_persisted_as_distinct_outcome(monkeypatch, tmp_path):
    class RateLimitedAdapter:
        def __init__(self, output_dir):
            self.output_dir = output_dir

        def generate(self, **kwargs):
            raise ProviderRateLimitError("OpenAI", "account rate exceeded")

    monkeypatch.setattr(evaluator_module, "EvalEngineAdapter", RateLimitedAdapter)
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    task = {
        "provider": "OpenAI",
        "model": "test-model",
        "full_prompt": "prompt in C major",
        "original_prompt": "prompt",
        "root": "C",
        "scale": "major",
        "use_thinking": False,
        "effort": None,
        "variation_name": "standard",
        "task_id": "task-rate-limit-0123456789abcdef-1",
    }
    run_path = tmp_path / "run"
    run_path.mkdir()

    result = evaluator._run_single(task, run_path, ["scale"])

    assert result["tests"]["overall_status"] == "rate_limited"
    assert result["tests"]["overall_pass"] is False


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


def test_run_tests_rejects_unknown_checks_before_running_any_check(monkeypatch, tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")

    def unexpected_scale_check(*args):
        raise AssertionError("scale check should not run")

    monkeypatch.setattr(
        evaluator,
        "AVAILABLE_TESTS",
        {**evaluator.AVAILABLE_TESTS, "scale": unexpected_scale_check},
    )

    with pytest.raises(ValueError, match=r"Unknown tests: not_a_check"):
        evaluator.run_tests(
            midi_data=MidiFile(),
            root="C",
            scale="major",
            prompt="melody",
            tests=["not_a_check"],
        )


def test_evaluate_rejects_unknown_checks_before_creating_run_directory(tmp_path):
    output_dir = tmp_path / "evaluations"
    evaluator = Evaluator(output_dir=output_dir)

    with pytest.raises(ValueError, match=r"Unknown tests: typo"):
        evaluator.evaluate(
            prompts="melody",
            roots=["C"],
            models=[],
            run_name="invalid-check",
            tests=["typo"],
        )

    assert not output_dir.exists()


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
        {
            "run_id": "20260723_000000_metric-contract_abc123",
            "timestamp": "20260723_000000",
            "run_name": "metric-contract",
        },
    )

    assert summary["run_id"] == "20260723_000000_metric-contract_abc123"
    assert summary["totals"]["total_cost"] == 0.25
    assert summary["totals"]["known_cost_generations"] == 1
    assert summary["totals"]["unknown_cost_generations"] == 1
    assert summary["totals"]["total_time"] == 9.0
    assert summary["totals"]["avg_successful_latency"] == 2.0
    assert summary["by_model"]["unknown"]["avg_latency"] is None


def test_summary_reports_latency_and_default_rates_for_ineligible_result(tmp_path):
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    summary = evaluator._generate_summary(
        [
            {
                "model": "model",
                "provider": "Ollama",
                "root": "C",
                "scale": "major",
                "metrics": {"api_latency": 2.0},
                "tests": {"overall_pass": False, "overall_status": "ineligible"},
                "error": None,
            }
        ],
        {"run_id": "ineligible-metrics"},
    )

    assert summary["by_model"]["model"]["avg_latency"] == 2.0
    assert summary["by_root"]["C"]["pass_rate"] == 0.0
    assert summary["by_scale"]["major"]["pass_rate"] == 0.0


def test_failed_generation_records_attempt_latency_and_contextual_log(monkeypatch, tmp_path):
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
        "full_prompt": "sensitive prompt",
        "original_prompt": "sensitive prompt",
        "root": "C",
        "scale": "major",
        "use_thinking": False,
        "effort": None,
        "variation_name": "standard",
        "task_id": "task-prompt-0123456789abcdef-1",
    }

    run_path = tmp_path / "run"
    run_path.mkdir()
    logger, handler = evaluator._create_run_logger(run_path, "failure")
    try:
        result = evaluator._run_single(task, run_path, ["scale"], logger)
    finally:
        evaluator._close_run_logger(logger, handler)

    assert result["error"] == "provider timed out"
    assert result["metrics"] == {
        "api_latency": None,
        "attempt_latency": 3.5,
        "cost": None,
        "cost_available": False,
    }
    log_contents = (run_path / "run.log").read_text(encoding="utf-8")
    assert "Task failed: task_id=task-prompt-0123456789abcdef-1" in log_contents
    assert "provider=OpenAI model=test-model root=C scale=major variation=standard" in log_contents
    assert "error_type=RuntimeError" in log_contents
    assert "Traceback:" in log_contents
    assert "sensitive prompt" not in log_contents


def test_run_log_excludes_task_success_telemetry(monkeypatch, tmp_path):
    class SuccessfulAdapter:
        def __init__(self, output_dir):
            self.output_dir = output_dir

        def generate(self, **kwargs):
            return MidiFile(), [{"role": "assistant", "content": "loop"}], 0.125

    monkeypatch.setattr("conductor_eval.evaluator.EvalEngineAdapter", SuccessfulAdapter)
    monkeypatch.setattr("conductor_eval.evaluator.time.perf_counter", lambda: next(clock))
    clock = iter([100.0, 101.25])
    evaluator = Evaluator(output_dir=tmp_path / "evaluations")
    monkeypatch.setattr(
        evaluator,
        "run_tests",
        lambda **kwargs: {"scale": {"passed": True}, "overall_pass": True},
    )
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
    run_path = tmp_path / "run"
    run_path.mkdir()
    logger, handler = evaluator._create_run_logger(run_path, "lifecycle")

    try:
        result = evaluator._run_single(task, run_path, ["scale"], logger)
    finally:
        evaluator._close_run_logger(logger, handler)

    log_contents = (run_path / "run.log").read_text(encoding="utf-8")
    assert result["metrics"]["api_latency"] == 1.25
    assert log_contents == ""

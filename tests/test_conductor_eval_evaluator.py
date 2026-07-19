import json
from types import SimpleNamespace

import pytest
from conductor_core import GenerationRequest
from mido import Message, MidiFile

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


def test_save_results_uses_per_result_loop_filename(tmp_path):
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
    }

    evaluator._save_results(result, midi, messages, run_path, task)

    result_dir = run_path / "results" / "OpenAI" / "gpt-test" / "warm_loop" / "C_major"
    assert (result_dir / "loop.mid").exists()
    legacy_filename = "output" + ".mid"
    assert not (result_dir / legacy_filename).exists()
    assert json.loads((result_dir / "messages.json").read_text(encoding="utf-8")) == messages


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

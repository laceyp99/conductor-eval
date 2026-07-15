import json

from conductor_eval.analysis import load_run


def _write_run(tmp_path, tests):
    run_path = tmp_path / "run"
    result_path = run_path / "results" / "OpenAI" / "model" / "prompt" / "C_major"
    result_path.mkdir(parents=True)
    (run_path / "config.json").write_text(
        json.dumps({"run_name": "analysis-test", "test_reasoning": False}),
        encoding="utf-8",
    )
    (result_path / "test_results.json").write_text(
        json.dumps(
            {
                "model": "model",
                "provider": "OpenAI",
                "prompt": "prompt in C major",
                "original_prompt": "prompt",
                "root": "C",
                "scale": "major",
                "config": {},
                "metrics": {},
                "tests": {"overall_pass": False, **tests},
            }
        ),
        encoding="utf-8",
    )
    return run_path


def test_load_run_flattens_new_check_results(tmp_path):
    run_path = _write_run(
        tmp_path,
        {
            "monophony": {
                "ran": True,
                "passed": False,
                "max_polyphony": 2,
                "polyphony_distribution": {"1": 3.0, "2": 1.0},
                "polyphony_percentages": {"1": 75.0, "2": 25.0},
            },
            "polyphony": {
                "ran": True,
                "passed": False,
                "max_polyphony": 2,
                "min_voices": 3,
                "params": {"min_voices": 3},
                "polyphony_distribution": {"2": 16.0},
                "polyphony_percentages": {"2": 100.0},
            },
            "chord_progression": {
                "ran": True,
                "passed": False,
                "params": {"progression": ["I", "V", "vi", "IV"]},
                "bars": [
                    {"bar": 1, "passed": True},
                    {
                        "bar": 2,
                        "passed": False,
                        "missing_pitch_classes": [7],
                        "extra_pitch_classes": [8],
                    },
                ],
            },
            "harmonic_rhythm": {
                "ran": True,
                "passed": False,
                "missing_onsets": [8.0],
                "unexpected_onsets": [6.0],
            },
            "chord_event_positions": {
                "ran": True,
                "passed": False,
                "missing_positions": [{"start_beat": 8.0, "end_beat": 12.0}],
                "unexpected_positions": [{"start_beat": 8.0, "end_beat": 10.0}],
            },
        },
    )

    df, _, _ = load_run(run_path)
    row = df.iloc[0]

    assert row["monophony_ran"]
    assert not row["monophony_pass"]
    assert row["monophony_max_polyphony"] == 2
    assert row["polyphony_min_voices"] == 3
    assert row["polyphony_voice_shortfall"] == 1
    assert [bar["bar"] for bar in row["chord_progression_failed_bars"]] == [2]
    assert row["harmonic_rhythm_missing_onsets"] == [8.0]
    assert row["harmonic_rhythm_unexpected_onsets"] == [6.0]
    assert row["chord_event_positions_missing"] == [{"start_beat": 8.0, "end_beat": 12.0}]
    assert row["chord_event_positions_unexpected"] == [{"start_beat": 8.0, "end_beat": 10.0}]


def test_load_run_treats_missing_and_not_run_checks_as_ineligible(tmp_path):
    run_path = _write_run(
        tmp_path,
        {
            "monophony": {"ran": False, "passed": True, "max_polyphony": 1},
            "scale": {"ran": True, "total": 1, "correct": 1, "incorrect": 0},
            "duration": {"ran": True, "total": 1, "correct": 1, "incorrect": 0},
        },
    )

    df, _, _ = load_run(run_path)
    row = df.iloc[0]

    assert not row["monophony_ran"]
    assert not row["monophony_pass"]
    assert not row["polyphony_ran"]
    assert not row["polyphony_pass"]
    assert not row["chord_progression_ran"]
    assert row["chord_progression_failed_bars"] == []
    assert row["scale_pass"]
    assert row["duration_pass"]

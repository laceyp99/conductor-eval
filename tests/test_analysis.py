import json
from html import unescape

import pandas as pd

from conductor_eval.analysis import (
    _build_combined_html,
    _format_pass_rate_summary,
    _overall_pass_rate_percent,
    build_chord_performance_by_model,
    build_cost_by_model,
    build_cost_vs_pass,
    build_duration_adherence_by_model,
    build_duration_errors_by_model,
    build_failure_rate_by_model,
    build_incorrect_intervals_by_model,
    build_incorrect_pitches_by_model,
    build_latency_box,
    build_latency_vs_pass,
    build_major_vs_minor_by_model,
    build_model_root_heatmap,
    build_pass_rate_by_model,
    build_reasoning_cost_effectiveness,
    build_reasoning_toggle_comparison,
    build_texture_performance_by_model,
    compute_scatter_label_layout,
    compute_text_positions,
    load_run,
)


def test_format_pass_rate_summary_uses_compact_outcome_text():
    assert _format_pass_rate_summary(4, 8, []) == (
        "50.0%",
        "4 / 8 eligible passed",
        "No generation errors",
        "#2ecc71",
    )
    assert _format_pass_rate_summary(0, 0, [("generation_error", 1)]) == (
        "N/A",
        "No eligible generations",
        "1 generation error",
        "#5dade2",
    )
    assert _format_pass_rate_summary(1, 4, [("rate_limited", 2)])[:3] == (
        "25.0%",
        "1 / 4 eligible passed",
        "2 rate limited",
    )
    assert (
        _format_pass_rate_summary(1, 2, [("ineligible", 2), ("check_error", 1)])[2]
        == "3 run exceptions"
    )


def test_metric_charts_exclude_unknown_costs_and_failed_latencies():
    df = pd.DataFrame(
        [
            {"model": "reported", "cost": 0.5, "api_latency": 2.0, "overall_pass": True},
            {"model": "unknown", "cost": None, "api_latency": None, "overall_pass": False},
        ]
    )

    assert list(build_cost_by_model(df).data[0].y) == ["reported"]
    assert [trace.name for trace in build_latency_box(df).data] == ["reported"]
    assert list(build_latency_vs_pass(df).data[0].customdata) == ["reported"]


def test_latency_vs_pass_uses_all_attempts_for_pass_rate():
    df = pd.DataFrame(
        [
            {"model": "alpha", "api_latency": 2.0, "overall_pass": True},
            {"model": "alpha", "api_latency": None, "overall_pass": False},
            {"model": "beta", "api_latency": 4.0, "overall_pass": False},
        ]
    )

    trace = build_latency_vs_pass(df).data[0]
    stats = {
        model: (latency, pass_rate)
        for model, latency, pass_rate in zip(trace.customdata, trace.x, trace.y, strict=True)
    }

    assert stats == {"alpha": (2.0, 50.0), "beta": (4.0, 0.0)}


def test_tradeoff_charts_include_ineligible_generation_telemetry():
    df = pd.DataFrame(
        [
            {
                "model": "alpha",
                "base_model": "alpha",
                "api_latency": 1.0,
                "cost": 1.0,
                "overall_pass": True,
            },
            {
                "model": "alpha",
                "base_model": "alpha",
                "api_latency": 9.0,
                "cost": 9.0,
                "overall_pass": None,
            },
        ]
    )

    latency = build_latency_vs_pass(df).data[0]
    cost = build_cost_vs_pass(df).data[0]
    reasoning = build_reasoning_cost_effectiveness(df).data[0]

    assert list(latency.x) == [5.0]
    assert list(latency.y) == [100.0]
    assert list(cost.x) == [5.0]
    assert list(cost.y) == [100.0]
    assert list(reasoning.x) == [5.0]
    assert list(reasoning.y) == [100.0]
    assert list(reasoning.customdata[0]) == ["alpha", 1, 1, 2]


def test_reasoning_toggle_uses_all_telemetry_and_only_eligible_pass_results():
    df = pd.DataFrame(
        [
            {
                "model": "alpha",
                "base_model": "alpha",
                "use_thinking": use_thinking,
                "effort": None,
                "api_latency": latency,
                "cost": cost,
                "overall_pass": overall_pass,
            }
            for use_thinking, latency, cost, overall_pass in [
                (False, 1.0, 1.0, True),
                (False, 9.0, 9.0, None),
                (True, 2.0, 2.0, False),
                (True, 10.0, 10.0, None),
            ]
        ]
    )

    figure = build_reasoning_toggle_comparison(df)

    assert list(figure.data[0].x) == [100.0]
    assert list(figure.data[1].x) == [0.0]
    assert list(figure.data[2].x) == [5.0]
    assert list(figure.data[3].x) == [6.0]
    assert list(figure.data[4].x) == [5.0]
    assert list(figure.data[5].x) == [6.0]


def test_overall_pass_rate_percent_defaults_to_zero_for_all_ineligible_results():
    df = pd.DataFrame([{"overall_pass": None}, {"overall_pass": None}])

    assert _overall_pass_rate_percent(df) == 0.0


def test_combined_html_escapes_run_metadata_and_preserves_unicode():
    run_name = "Résumé <script>alert(\"run\" & 'name')</script>"
    timestamp = '2026-07-19 </p><script>alert("timestamp")</script>'
    df = pd.DataFrame([{"model": "model", "overall_pass": True, "cost": 0.0}])

    combined_html = _build_combined_html({}, run_name, timestamp, {}, df)

    assert '<script>alert("run"' not in combined_html
    assert '</p><script>alert("timestamp")</script>' not in combined_html
    assert "&lt;script&gt;" in combined_html
    assert "&lt;/p&gt;&lt;script&gt;" in combined_html
    assert "&quot;run&quot; &amp; &#x27;name&#x27;" in combined_html
    assert run_name in unescape(combined_html)
    assert timestamp in unescape(combined_html)


def test_combined_html_reports_known_cost_count():
    df = pd.DataFrame(
        [
            {"model": "reported", "overall_pass": True, "cost": 0.5},
            {"model": "unknown", "overall_pass": False, "cost": None},
        ]
    )

    combined_html = _build_combined_html({}, "run", "timestamp", {}, df)

    assert "Total Reported Cost" in combined_html
    assert "$0.5000" in combined_html
    assert "1/2 costs reported" in combined_html


def _write_run(tmp_path, tests):
    run_path = tmp_path / "run"
    result_path = run_path / "results" / "task-baseline-0123456789abcdef-1"
    result_path.mkdir(parents=True)
    (run_path / "config.json").write_text(
        json.dumps({"run_name": "analysis-test", "test_reasoning": False}),
        encoding="utf-8",
    )
    (result_path / "test_results.json").write_text(
        json.dumps(
            {
                "task_id": "task-baseline-0123456789abcdef-1",
                "model": "model",
                "provider": "OpenAI",
                "prompt": "prompt in C major",
                "original_prompt": "prompt",
                "root": "C",
                "scale": "major",
                "config": {"variation_name": "standard"},
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


def test_load_run_uses_persisted_metadata_with_arbitrary_task_directory_names(tmp_path):
    run_path = _write_run(tmp_path, {})
    result_path = run_path / "results" / "this-directory-name-has-no-meaning"
    result_path.mkdir(parents=True)
    (result_path / "test_results.json").write_text(
        json.dumps(
            {
                "task_id": "task-readable-0123456789abcdef-1",
                "model": "new-model",
                "provider": "OpenAI",
                "config": {"variation_name": "high"},
                "metrics": {},
                "tests": {"overall_pass": True},
            }
        ),
        encoding="utf-8",
    )

    df, _, _ = load_run(run_path)

    assert df.loc[df["model"] == "new-model", "variation"].item() == "high"
    assert df.loc[df["model"] == "new-model", "task_id"].item() == (
        "task-readable-0123456789abcdef-1"
    )


def test_load_run_does_not_infer_variation_from_task_directory_name(tmp_path):
    run_path = _write_run(tmp_path, {})
    result_path = run_path / "results" / "definitely-not-a-variation"
    result_path.mkdir(parents=True)
    (result_path / "test_results.json").write_text(
        json.dumps(
            {
                "model": "missing-variation",
                "provider": "OpenAI",
                "config": {},
                "metrics": {},
                "tests": {"overall_pass": True},
            }
        ),
        encoding="utf-8",
    )

    df, _, _ = load_run(run_path)

    assert df.loc[df["model"] == "missing-variation", "variation"].item() == "unknown"


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


def test_texture_performance_excludes_checks_without_completed_note_evidence():
    df = pd.DataFrame(
        [
            {
                "model": "model",
                "monophony_ran": True,
                "monophony_eligible": False,
                "monophony_pass": False,
                "polyphony_ran": False,
                "polyphony_eligible": False,
                "polyphony_pass": False,
            }
        ]
    )

    figure = build_texture_performance_by_model(df)

    assert not figure.data


def test_load_run_excludes_ineligible_results_from_overall_pass_rates(tmp_path):
    run_path = _write_run(
        tmp_path,
        {
            "overall_status": "ineligible",
            "scale": {"ran": True, "total": 0, "correct": 0, "incorrect": 0},
            "duration": {"ran": False, "skipped": "No duration keyword detected in prompt"},
        },
    )

    df, _, _ = load_run(run_path)
    row = df.iloc[0]

    assert row["overall_status"] == "ineligible"
    assert row["overall_pass"] is None
    assert not row["scale_pass"]
    assert not row["duration_pass"]
    assert not build_pass_rate_by_model(df).data


def test_load_run_excludes_check_errors_from_overall_pass_rates(tmp_path):
    run_path = _write_run(
        tmp_path,
        {
            "overall_status": "check_error",
            "scale": {
                "ran": False,
                "eligible": False,
                "status": "check_error",
                "error": "checker failed",
            },
        },
    )

    df, _, _ = load_run(run_path)
    row = df.iloc[0]

    assert row["overall_status"] == "check_error"
    assert row["overall_pass"] is None
    assert not row["overall_eligible"]
    assert not build_pass_rate_by_model(df).data


def test_load_run_preserves_rate_limited_outcome(tmp_path):
    run_path = _write_run(tmp_path, {"overall_status": "rate_limited"})

    df, _, _ = load_run(run_path)
    row = df.iloc[0]

    assert row["overall_status"] == "rate_limited"
    assert row["overall_pass"] is None
    assert not row["overall_eligible"]
    assert not build_pass_rate_by_model(df).data


def test_duration_adherence_groups_by_note_length_and_excludes_not_run_rows():
    df = pd.DataFrame(
        [
            {
                "model": "alpha",
                "duration_ran": True,
                "duration_param": "quarter",
                "duration_pass": True,
            },
            {
                "model": "alpha",
                "duration_ran": False,
                "duration_param": "quarter",
                "duration_pass": False,
            },
            {
                "model": "alpha",
                "duration_ran": True,
                "duration_param": "eighth",
                "duration_pass": False,
            },
            {
                "model": "beta",
                "duration_ran": True,
                "duration_param": "quarter",
                "duration_pass": False,
            },
        ]
    )

    fig = build_duration_adherence_by_model(df)

    assert [trace.name for trace in fig.data] == ["Quarter Notes", "Eighth Notes"]
    quarter = fig.data[0]
    assert list(quarter.x) == ["alpha", "beta"]
    assert list(quarter.y) == [100.0, 0.0]
    assert [list(counts) for counts in quarter.customdata] == [[1, 1], [0, 1]]
    assert list(fig.data[1].y) == [0.0]


def test_texture_performance_uses_each_checks_eligible_rows():
    df = pd.DataFrame(
        [
            {
                "model": "alpha",
                "monophony_ran": True,
                "monophony_pass": True,
                "polyphony_ran": False,
                "polyphony_pass": False,
            },
            {
                "model": "beta",
                "monophony_ran": False,
                "monophony_pass": False,
                "polyphony_ran": True,
                "polyphony_pass": False,
            },
        ]
    )

    fig = build_texture_performance_by_model(df)

    assert [trace.name for trace in fig.data] == ["Monophony", "Polyphony"]
    assert list(fig.data[0].x) == ["alpha"]
    assert list(fig.data[0].y) == [100.0]
    assert list(fig.data[1].x) == ["beta"]
    assert list(fig.data[1].y) == [0.0]


def test_chord_performance_has_empty_state_when_no_chord_checks_ran():
    df = pd.DataFrame(
        [
            {
                "model": "alpha",
                "chord_progression_ran": False,
                "chord_progression_pass": False,
                "harmonic_rhythm_ran": False,
                "harmonic_rhythm_pass": False,
                "chord_event_positions_ran": False,
                "chord_event_positions_pass": False,
            }
        ]
    )

    fig = build_chord_performance_by_model(df)

    assert not fig.data
    assert fig.layout.annotations[0].text == "No chord checks ran"


def test_rate_charts_share_passed_and_generation_count_context():
    df = pd.DataFrame(
        [
            {"model": "alpha", "root": "C", "scale": "major", "overall_pass": True},
            {"model": "alpha", "root": "D", "scale": "minor", "overall_pass": False},
            {"model": "beta", "root": "C", "scale": "major", "overall_pass": False},
        ]
    )

    overall = build_pass_rate_by_model(df)
    major_minor = build_major_vs_minor_by_model(df)
    heatmap = build_model_root_heatmap(df)

    alpha_index = list(overall.data[0].y).index("alpha")
    assert list(overall.data[0].customdata[alpha_index]) == [1, 2]
    assert "Passed: %{customdata[0]}" in overall.data[0].hovertemplate
    assert list(major_minor.data[0].customdata[0]) == [1, 1]
    assert "Eligible generations: %{customdata[1]}" in major_minor.data[0].hovertemplate
    assert list(heatmap.data[0].customdata[0][0]) == [1, 1]


def test_failure_rate_uses_error_and_generation_count_labels():
    df = pd.DataFrame(
        [
            {"model": "alpha", "has_error": True},
            {"model": "alpha", "has_error": False},
        ]
    )

    fig = build_failure_rate_by_model(df)

    assert list(fig.data[0].customdata[0]) == [1, 2]
    assert fig.data[0].text[0] == "50.0% (1/2)"
    assert "Errors: %{customdata[0]}" in fig.data[0].hovertemplate


def test_tradeoff_and_total_cost_hovers_omit_unneeded_count_context():
    df = pd.DataFrame(
        [
            {"model": "alpha", "api_latency": 2.0, "cost": 0.25, "overall_pass": True},
            {"model": "alpha", "api_latency": 4.0, "cost": 0.75, "overall_pass": False},
        ]
    )

    latency = build_latency_vs_pass(df).data[0].hovertemplate
    cost_tradeoff = build_cost_vs_pass(df).data[0].hovertemplate
    total_cost = build_cost_by_model(df).data[0].hovertemplate

    assert "Pass rate: %{y:.1f}%" in latency
    assert "Passed:" not in latency
    assert "Generations:" not in latency
    assert "Pass rate: %{y:.1f}%" in cost_tradeoff
    assert "Passed:" not in cost_tradeoff
    assert "Generations:" not in cost_tradeoff
    assert "Cost per generation:" in total_cost
    assert "Cost per success:" not in total_cost
    assert "Passed:" not in total_cost
    assert "Generations:" not in total_cost


def test_scatter_text_positions_point_inward_at_horizontal_boundaries():
    positions = compute_text_positions([0.0, 0.5, 1.0], [50.0, 50.0, 50.0])

    assert positions[0].endswith("right")
    assert positions[-1].endswith("left")


def test_scatter_text_positions_point_inward_at_vertical_boundaries():
    positions = compute_text_positions(
        [0.5, 0.5, 0.5],
        [0.0, 50.0, 100.0],
        y_bounds=(0, 105),
    )

    assert positions[0].startswith("top")
    assert positions[-1].startswith("bottom")


def test_scatter_text_position_handles_single_point_at_top_boundary():
    positions = compute_text_positions([0.5], [100.0], y_bounds=(0, 105))

    assert positions == ["bottom center"]


def test_scatter_label_layout_stays_in_bounds_without_collisions():
    x_values = [0.008, 0.0085, 0.009, 0.0095, 0.010, 0.018, 0.0182, 0.019, 0.022, 0.023]
    y_values = [80, 90, 100, 100, 100, 60, 60, 80, 90, 100]
    labels = [f"gpt-5.6-model ({effort})" for effort in range(len(x_values))]
    layouts = compute_scatter_label_layout(
        x_values,
        y_values,
        labels,
        x_bounds=(0.005, 0.025),
        y_bounds=(0, 105),
        plot_width=520,
    )

    rectangles = [layout["rect"] for layout in layouts]
    assert all(
        left >= 0 and right <= 1 and bottom >= 0 and top <= 1
        for left, right, bottom, top in rectangles
    )
    for index, first in enumerate(rectangles):
        for second in rectangles[index + 1 :]:
            assert (
                first[1] <= second[0]
                or second[1] <= first[0]
                or first[3] <= second[2]
                or second[3] <= first[2]
            )


def test_tradeoff_charts_compact_long_labels_and_keep_full_names_in_hover():
    long_model = "provider/model-with-an-unusually-long-version-identifier"
    df = pd.DataFrame(
        [
            {
                "model": long_model,
                "api_latency": 1.0,
                "cost": 0.01,
                "overall_pass": True,
            },
            {
                "model": "short-model",
                "api_latency": 2.0,
                "cost": 0.02,
                "overall_pass": False,
            },
        ]
    )

    for figure in (build_latency_vs_pass(df), build_cost_vs_pass(df)):
        trace = figure.data[0]
        long_model_index = list(trace.customdata).index(long_model)
        compact_annotation = next(
            annotation for annotation in figure.layout.annotations if "…" in annotation.text
        )

        assert trace.mode == "markers"
        assert len(compact_annotation.text) == 24
        assert len(figure.layout.annotations) == 2
        assert trace.customdata[long_model_index] == long_model
        assert "Model: %{customdata}" in trace.hovertemplate


def test_pitch_and_interval_error_hovers_include_musical_context_and_share():
    df = pd.DataFrame(
        [
            {
                "model": "alpha",
                "root": "C",
                "scale_ran": True,
                "scale_pitches_incorrect": [1, 1, 3],
            }
        ]
    )

    pitches = build_incorrect_pitches_by_model(df).data[0]
    intervals = build_incorrect_intervals_by_model(df).data[0]

    assert "Pitch class: %{customdata[0]}" in pitches.hovertemplate
    assert "Occurrences: %{y}" in pitches.hovertemplate
    assert sorted(value[1] for value in pitches.customdata) == [33.3, 66.7]
    assert "Semitones from prompted root: %{customdata[0]}" in intervals.hovertemplate
    assert "Occurrences: %{y}" in intervals.hovertemplate
    assert sorted(value[1] for value in intervals.customdata) == [33.3, 66.7]


def test_duration_error_hover_includes_count_and_within_model_share():
    df = pd.DataFrame(
        [
            {
                "model": "alpha",
                "duration_ran": True,
                "duration_param": "quarter",
                "duration_lengths": {"0.5": 3, "2.0": 1},
            }
        ]
    )

    trace = build_duration_errors_by_model(df).data[0]

    assert "Incorrect notes: %{y}" in trace.hovertemplate
    assert "Share of this model's duration errors" in trace.hovertemplate
    assert sorted(trace.customdata) == [25.0, 75.0]


def test_latency_distribution_hover_omits_fence_statistics():
    df = pd.DataFrame(
        [
            {"model": "alpha", "api_latency": 1.0},
            {"model": "alpha", "api_latency": 2.0},
            {"model": "alpha", "api_latency": 3.0},
        ]
    )

    hover = build_latency_box(df).data[0].hovertemplate.lower()

    assert "latency: %{y:.2f}s" in hover
    assert "upper fence" not in hover
    assert "lower fence" not in hover


def test_model_variants_share_base_model_and_effort_order_across_charts():
    models = [
        "zeta (medium)",
        "alpha (high)",
        "alpha (none)",
        "zeta (low)",
        "alpha (max)",
        "alpha (low)",
        "alpha (medium)",
    ]
    df = pd.DataFrame(
        [
            {
                "model": model,
                "root": "C",
                "overall_pass": index % 2 == 0,
                "api_latency": float(index + 1),
                "cost": float(index + 1),
                "has_error": index % 2 == 1,
            }
            for index, model in enumerate(models)
        ]
    )
    expected = [
        "alpha (none)",
        "alpha (low)",
        "alpha (medium)",
        "alpha (high)",
        "alpha (max)",
        "zeta (low)",
        "zeta (medium)",
    ]

    assert [trace.name for trace in build_latency_box(df).data] == expected
    pass_rate = build_pass_rate_by_model(df)
    assert list(pass_rate.data[0].y) == expected
    assert pass_rate.layout.yaxis.autorange == "reversed"
    assert list(build_cost_by_model(df).data[0].y) == expected
    assert list(build_failure_rate_by_model(df).data[0].y) == expected
    heatmap = build_model_root_heatmap(df)
    assert list(heatmap.data[0].y) == expected
    assert heatmap.layout.yaxis.autorange == "reversed"


def test_model_variant_order_has_deterministic_fallbacks():
    models = [
        "beta",
        "alpha (turbo)",
        "alpha (low)",
        "alpha",
        "alpha (experimental)",
        "alpha (none)",
    ]
    df = pd.DataFrame(
        [{"model": model, "api_latency": float(index)} for index, model in enumerate(models)]
    )

    assert [trace.name for trace in build_latency_box(df).data] == [
        "alpha",
        "alpha (none)",
        "alpha (low)",
        "alpha (experimental)",
        "alpha (turbo)",
        "beta",
    ]

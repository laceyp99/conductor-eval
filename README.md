<div align="center">
  <img src="app/readme-logo.png" alt="Conductor Eval Logo" width="50%">
</div>

A unified evaluation framework for testing MIDI loop generation across multiple AI models, with an interactive Plotly Dash dashboard for analyzing results.

This framework is not a replacement for its local pytest suite. Use pytest for
fast local checks of deterministic Eval code paths. Use the evaluator when you
want to measure prompt-to-model behavior, musical validity, latency, cost, or
reasoning-mode differences across real models.

## Installation

### Develop with uv

Eval uses [uv](https://docs.astral.sh/uv/) 0.11.16 or newer for its primary
development workflow. From the `conductor-eval` project directory, create the
locked environment with the dashboard and development extras:

```powershell
uv sync --locked --all-extras
```

You do not need to activate the environment. Run the project checks with:

```powershell
uv run --locked --all-extras ruff format --check .
uv run --locked --all-extras ruff check .
uv run --locked --all-extras pytest -q
uv build
```

When intentionally updating dependencies, run `uv lock --upgrade`, review the
lockfile diff, and rerun the checks. Do not edit `uv.lock` by hand.

### Install with pip

Eval remains a standard setuptools package for contributors and consumers who
do not use uv. On Windows, use the virtual environment's interpreter explicitly:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dashboard,dev]"
```

Use `.\.venv\Scripts\python.exe -m pip install .` instead for a non-editable
base installation without the dashboard or development tools.

The uv lockfile does not constrain pip installations; pip resolves versions
from the compatible ranges and pins declared in `pyproject.toml`.

Key packages: `dash`, `dash-bootstrap-components`, `pandas`, `plotly`, `mido`, `rich`.

## Quick Start

### Run an Evaluation

```python
from conductor_eval import Evaluator

evaluator = Evaluator(temperature=0.0)

results = evaluator.evaluate(
    prompts="an arpeggiator using only quarter notes",
    roots=["C", "G", "D"],
    models="openai",
    run_name="my_first_eval",
)
```

By default, Eval stores mutable data under
`~/.conductor/eval/evaluations/`. This location is independent of the checkout,
current working directory, virtual environment, and installed package path.

### Data Directory

Conductor projects share `~/.conductor` as their default suite root. Eval owns
the stable `eval` project directory beneath that root:

```text
~/.conductor/
└── eval/
    └── evaluations/
        └── <timestamp>_<run-name>_<uuid>/
            └── run.log
```

The directory precedence, from highest to lowest, is:

1. `CONDUCTOR_EVAL_HOME`: the complete Eval data directory.
2. `CONDUCTOR_HOME`: the suite root; Eval appends `eval`.
3. `Path.home() / ".conductor" / "eval"`.

Both environment-variable paths support `~` expansion. An explicit
`Evaluator(output_dir=...)` remains narrower than these defaults and writes to
the exact directory supplied.

PowerShell examples:

```powershell
# Put every Conductor project's data beneath another suite root.
$env:CONDUCTOR_HOME = "D:\ConductorData"
# Eval now defaults to D:\ConductorData\eval\evaluations

# Or override Eval alone with its complete project data directory.
$env:CONDUCTOR_EVAL_HOME = "D:\ConductorEvalData"
# Eval now defaults to D:\ConductorEvalData\evaluations
```

Path resolution and package import do not create directories. Eval creates
them only when it initializes its existing file logging or writes a run. This
change does not move, overwrite, or delete existing checkout-local
`evaluations/` or `runs/` directories; pass one explicitly as `output_dir` to
continue portable or legacy operation.

Evaluation data can be large because every run retains Core generation
artifacts and also copies report MIDI and messages; dashboard exports add HTML
files. Storage grows with prompts × roots × two scales × model variants. Review
and remove complete, unneeded runs manually; Eval performs no automatic
migration or cleanup.

### Direct Script Safeguard

Running `uv run --locked --all-extras python -m conductor_eval.evaluator`
directly is guarded because the
example in that file starts a broad cloud evaluation across multiple paid
providers. The script prints a warning and requires the exact confirmation
phrase `RUN CLOUD EVALUATION` before it creates an `Evaluator` or starts any
provider calls.

Pressing Enter, sending no input, or typing anything else aborts the script
without creating evaluation outputs. For smaller intentional runs, prefer the
Python API examples above so you can choose the prompts, roots, models, and run
name explicitly.

### Launch the Dashboard

```powershell
# Interactive run selection
uv run --locked --all-extras python -m conductor_eval.analysis

# Direct path to a run
uv run --locked --all-extras python -m conductor_eval.analysis "$HOME\.conductor\eval\evaluations\20260210_224954_arpeggiator_local"
```

The dashboard opens at `http://127.0.0.1:8050/`. Pip users can run the same
module with `.\.venv\Scripts\python.exe` instead of the `uv run ... python`
prefix.

---

## Evaluator

### Basic Evaluation

```python
from conductor_eval import Evaluator

evaluator = Evaluator(temperature=0.0)

# Single prompt, multiple roots, one provider
results = evaluator.evaluate(
    prompts="an arpeggiator using only quarter notes",
    roots=["C", "G"],
    models="openai",
    run_name="quarter_note_test",
)
```

The evaluator automatically appends `" in {root} {scale}"` to each prompt and runs both major and minor scales for every root.

### Multiple Prompts

Test different musical patterns in a single run:

```python
results = evaluator.evaluate(
    prompts=[
        "an arpeggiator using only quarter notes",
        "an arpeggiator using only eighth notes",
        "an arpeggiator using only sixteenth notes",
    ],
    roots=["C", "D", "E", "F", "G", "A", "B"],
    models="all",
    run_name="duration_comparison",
)
```

### Model Selection

The `models` parameter accepts several formats:

| Value | Description |
|-------|-------------|
| `"all"` | All models from all providers (cloud + Ollama) |
| `"openai"` | All OpenAI models |
| `"anthropic"` | All Anthropic models |
| `"google"` | All Google Gemini models |
| `"ollama"` | All local Ollama models |
| `["gpt-5", "claude-sonnet-4-6"]` | Specific models by name |

Eval uses provider metadata to organize batches and reports, but generation
requests identify only the selected model. Conductor Core resolves the actual
provider route and records that provider with its generation artifacts.

### Cloud Rate Scheduling

Eval spaces cloud request starts independently for each `(provider, base_model)`.
The first request for a model may start immediately; later starts are separated
by at least `60 / RPM` seconds measured with a monotonic clock. The next slot is
always calculated from the actual previous Core-generation start, so delayed
work never causes a catch-up burst. Prompts, roots, scales, and reasoning
variations for one base model share the same FIFO schedule, while other model
queues can advance independently.

Core metadata supplies the default RPM. Account-specific limits can be recorded
for one run with nested provider/model overrides:

```python
results = evaluator.evaluate(
    prompts="an arpeggiator using only quarter notes",
    roots=["C", "G"],
    models=["gpt-5", "claude-sonnet-4-5"],
    run_name="paced_account_tier",
    rpm_overrides={
        "OpenAI": {"gpt-5": 60},
        "Anthropic": {"claude-sonnet-4-5": 40},
    },
    per_model_concurrency=5,
    global_cloud_concurrency=25,
)
```

Override precedence is run override, then Core baseline, otherwise failure.
RPM and concurrency values must be positive integers; booleans, strings,
decimals, zero, and negative values are rejected. Unknown providers/models,
duplicate case-insensitive provider keys, and overrides for unselected models
are also rejected. Model resolution and all of this validation happen before a
run directory or provider request is created.

RPM pacing is separate from concurrency. `per_model_concurrency` defaults to
`5`, allowing slow requests for one model to overlap, while
`global_cloud_concurrency` defaults to `25` and bounds all cloud work. Both are
upper bounds.

### Testing Reasoning Variations

When `test_reasoning=True`, the evaluator tests all thinking modes and effort levels for compatible models:

```python
results = evaluator.evaluate(
    prompts="complex chord progression",
    roots=["C", "G"],
    models=["o3", "claude-sonnet-4-5"],
    run_name="reasoning_test",
    test_reasoning=True,
)
```

| Provider | Model Type | Variations |
|----------|------------|------------|
| OpenAI | `gpt-5.x` and `o`-series reasoning models | effort levels only; current families use `none` to `xhigh`, `minimal` to `high`, or `low` to `high` depending on model |
| Anthropic | Claude 4.x reasoning models | either effort levels only for Claude 5 models, `claude-opus-4-8`, `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6`, or `standard` / `w_reasoning` toggle for other reasoning-capable Claude 4.x models |
| Google | Gemini 3.x and 2.5 reasoning models | Gemini 3.x uses effort levels; Gemini 2.5 models use `standard` / `w_reasoning` toggle |
| Ollama | All | standard only in the evaluator |

### Configuring Tests

The `tests` parameter controls which validation tests run on generated MIDI:

```python
results = evaluator.evaluate(
    prompts="an arpeggiator using only quarter notes",
    roots=["C"],
    models="openai",
    run_name="scale_only_test",
    tests=["scale"],  # Only run scale test, skip duration
)
```

| Test | Description | Auto-Detection |
|------|-------------|----------------|
| `scale` | Validates notes belong to the specified scale | Always uses root/scale from prompt |
| `duration` | Validates note durations match expected value | Detects from keywords: `quarter`, `eighth`, `sixteenth`, `16th`, `8th`, `half`, `whole` |
| `monophony` | Validates that completed notes never overlap | None; no parameters required |
| `polyphony` | Validates that the MIDI reaches a minimum number of simultaneous voices | None; set `min_voices` in `test_params` (default: `2`) |
| `chord_progression` | Validates diatonic chord pitch classes at fixed harmonic boundaries | None; set `progression` and optional `beats_per_chord`/`strict` in `test_params`; root and scale are supplied automatically |
| `harmonic_rhythm` | Validates that completed notes begin only at the expected beat positions | None; set `expected_onsets` in `test_params` |
| `chord_event_positions` | Validates that every completed note uses an expected start/end beat pair | None; set `expected_starts` and `expected_ends` in `test_params` |

The `scale` test always runs since root and scale are always applied to prompts. Duration keywords are owned by `conductor_core.music` as `DURATION_KEYWORDS` and shared with the evaluator. Parameters for the other tests are passed through `test_params`, keyed by test name.

### Output Structure

Each evaluation run creates a timestamped, unique directory beneath Eval's data
directory (shown here with the default suite root):

```
~/.conductor/eval/evaluations/
└── 20260210_224954_123456_my_first_eval-<hash>_<uuid16>/
    ├── run.log                        # Eval-owned lifecycle and error log
    ├── config.json                    # Full evaluation configuration
    ├── task_manifest.json             # Initial/final task-state snapshot
    ├── task_events.jsonl              # Durable append-only execution journal
    ├── summary.json                   # Aggregated results + statistics
    ├── core_artifacts/                # Core-owned MIDI, messages, and metadata
    ├── analysis/                      # Created by dashboard export
    │   └── dashboard.html
    └── results/
        └── task-an_arpeggiator_using-<fingerprint>-1/
            ├── loop.mid           # Generated MIDI file
            ├── messages.json      # Chat history (for fine-tuning)
            └── test_results.json  # Individual test results and task metadata
```

The evaluator intentionally retains `core_artifacts/` after copying MIDI and
messages into the report-oriented `results/` tree. Core owns generation
persistence, and retaining its canonical artifacts preserves provenance and
provider metadata for debugging. Eval does not selectively delete those files;
remove an entire completed run externally when its artifacts are no longer
needed.

Run directories include microseconds, a hash-backed 32-character run-name
component, and a 16-character UUID suffix. Each result is stored directly
beneath `results/` in a directory named
`task-<32-character sanitized prompt>-<16-character fingerprint>-<occurrence>`.
The fingerprint covers all task inputs and the occurrence always starts at
`1`, so repeated tasks remain distinct. A run or task directory collision
fails instead of overwriting artifacts. Result JSON metadata is authoritative;
the analysis loader does not infer meaning from directory names. Each run owns
one non-propagating `run.log`, which records run start and completion plus
contextual task and run failures. Eval does not capture prompts, provider
payloads, or host/root logger output.

When using `test_reasoning`, each variation receives its own task directory;
the variation is recorded in `test_results.json` rather than a subfolder:

```
# With test_reasoning=True
results/
├── task-an_arpeggiator_using-<fingerprint-none>-1/    # variation: none
├── task-an_arpeggiator_using-<fingerprint-low>-1/     # variation: low
├── task-an_arpeggiator_using-<fingerprint-medium>-1/  # variation: medium
├── task-an_arpeggiator_using-<fingerprint-high>-1/    # variation: high
└── task-an_arpeggiator_using-<fingerprint-xhigh>-1/   # variation: xhigh
```

#### config.json

Stores the full configuration used for the run:

```json
{
    "run_name": "my_first_eval",
    "timestamp": "20260207_143022_123456",
    "run_id": "20260207_143022_123456_my_first_eval-<hash>_<uuid16>",
    "prompts": ["an arpeggiator using only quarter notes"],
    "roots": ["C", "G"],
    "scales": ["major", "minor"],
    "models": [["OpenAI", "gpt-5"]],
    "tests": ["scale", "duration"],
    "test_reasoning": false,
    "temperature": 0.0,
    "rate_limits": {
        "per_model_concurrency": 5,
        "global_cloud_concurrency": 25,
        "models": [
            {
                "provider": "OpenAI",
                "model": "gpt-5",
                "baseline_rpm": 500,
                "override_rpm": null,
                "effective_rpm": 500,
                "source": "core"
            }
        ]
    }
}
```

#### task_manifest.json

The versioned manifest is atomically written with every task in `queued` state
before dispatch begins. Each entry separates an immutable `spec` (provider,
model, prompts, musical inputs, reasoning settings, tests, temperature, and
effective RPM) from mutable `execution` state.

During execution, state changes are appended as compact, monotonically
sequenced records in `task_events.jsonl`. Each append is flushed and synced to
disk without rewriting every task specification. Readers materializing a live
or interrupted run apply complete journal records newer than the manifest's
`last_event_sequence`; an incomplete final line from an interrupted write is
ignored. A successfully completed run atomically replaces `task_manifest.json`
with the fully materialized final state. The journal is retained for recovery
and auditing, and records already included in the final snapshot are skipped
during replay.

Execution states are `queued`, `dispatched`, `completed`, `failed`, `throttled`,
and `unstarted_due_to_throttling`. Unstarted entries link to the triggering
throttled task and never receive synthetic result directories. The manifest
contains prompts because they are required task inputs, but excludes provider
responses, generated messages, credentials, and environment values. It records
enough immutable state for a future explicit resume feature; this release does
not execute resumed runs.

#### summary.json

Aggregated statistics for the entire run:

```json
{
    "run_id": "20260207_143022_123456_my_first_eval-<hash>_<uuid16>",
    "totals": {
        "total_generations": 48,
        "planned_tasks": 51,
        "dispatched_tasks": 48,
        "completed_tasks": 45,
        "generation_failure_tasks": 2,
        "throttled_generation_tasks": 1,
        "unstarted_due_to_throttling_tasks": 3,
        "successful_generations": 45,
        "failed_generations": 3,
        "generation_error_generations": 3,
        "throttled_generations": 1,
        "check_error_generations": 0,
        "validation_failed_generations": 6,
        "ineligible_generations": 3,
        "eligible_generations": 42,
        "overall_pass_count": 36,
        "overall_pass_rate": 0.857,
        "total_cost": 1.25,
        "total_time": 120.5
    },
    "by_model": {
        "gpt-5": {
            "provider": "OpenAI",
            "tested": 24,
            "passed": 20,
            "pass_rate": 0.833,
            "total_cost": 0.50,
            "avg_latency": 2.1
        }
    },
    "by_root": { "C": { "tested": 24, "passed": 18, "pass_rate": 0.75 } },
    "by_scale": { "major": { "tested": 24, "passed": 20, "pass_rate": 0.833 } }
}
```

#### test_results.json

Individual results for each generation:

```json
{
    "task_id": "task-an_arpeggiator_using-<fingerprint>-1",
    "model": "gpt-5",
    "provider": "OpenAI",
    "prompt": "an arpeggiator using only quarter notes in C major",
    "original_prompt": "an arpeggiator using only quarter notes",
    "root": "C",
    "scale": "major",
    "config": {
        "use_thinking": false,
        "effort": null,
        "temperature": 0.0,
        "variation_name": "standard"
    },
    "metrics": {
        "api_latency": 2.34,
        "cost": 0.0025
    },
    "tests": {
        "scale": {
            "ran": true,
            "params": { "root": "C", "scale": "major" },
            "total": 16,
            "correct": 16,
            "incorrect": 0,
            "eligible": true,
            "status": "passed",
            "pitches": { "correct": [0, 2, 4, 5, 7, 9, 11], "incorrect": [] }
        },
        "duration": {
            "ran": true,
            "params": { "duration": "quarter" },
            "detected_from_prompt": true,
            "total": 16,
            "correct": 16,
            "incorrect": 0,
            "lengths": {}
        },
        "overall_pass": true,
        "overall_status": "passed"
    }
}
```

Overall pass rates use only eligible results as their denominator. Each result persists
an `overall_status` of `passed`, `failed`, `ineligible`, `generation_error`, or
`check_error`. A check with no examined notes is ineligible and can never make
`overall_pass` true. A checker exception is a `check_error`, is excluded from pass-rate
denominators, and is reported separately from a musical validation failure.
Typed provider rate-limit failures remain generation results with
`error_type: "ProviderRateLimitError"` and
`failure_kind: "provider_rate_limit"`; they are operational failures and never
enter musical pass-rate denominators.


## Analysis
### Global Filters

A filter bar at the top of every page lets you narrow results by:

- **Models** -- Select which models to include
- **Root Notes** -- Filter by root note (e.g. C, F#, Eb)
- **Scale Type** -- Major, minor, or both
- **Variation** -- Standard and reasoning effort levels

All charts update in real time when filters change.

### Dashboard Tabs

#### Tab 1: Overview
- Metric cards: total generations, pass rate, best/worst model, total cost, average latency
- Overall pass rate by model (horizontal bar chart)

#### Tab 2: Model Performance
- Per-test breakdown: scale vs duration vs overall pass rate per model
- Major vs minor pass rate comparison per model
- Model x scale heatmap
- Model x root heatmap

#### Tab 3: Root & Scale
- Pass rate by root note
- Major vs minor pass rate per root note
- Full model x root+scale heatmap

#### Tab 4: Latency
- Latency distribution box plots per model
- Latency vs pass rate scatter plot

#### Tab 5: Cost
- Total cost by model
- Cost per generation vs pass rate scatter plot

#### Tab 6: Reasoning *(only shown when `test_reasoning=True`)*
- Effort impact delta: pass rate change across effort levels per model
- Reasoning toggle comparison: pass rate with thinking on vs off for toggle-based models
- Reasoning cost-effectiveness: cost vs pass rate scatter colored by effort level

#### Tab 7: Error Patterns
- Generation failure rate (API/conversion errors) per model
- Most common incorrect pitch classes by model (as note names)
- Incorrect intervals relative to prompted root per model (e.g. m3, P5 -- helps identify systematic confusions)
- Incorrect durations by model showing actual vs requested duration

### Exporting

Click the **Export Dashboard** button to save all charts as individual HTML files plus a combined `dashboard.html` to `<evaluations-dir>/<run>/analysis/`. The number of exported charts depends on the run's features (16 base charts, plus 3 for reasoning when applicable):

```
~/.conductor/eval/evaluations/20260210_224954_arpeggiator_local/
└── analysis/
    ├── dashboard.html              # Combined single-page dashboard
    ├── pass_rate_by_model.html
    ├── per_test_breakdown.html
    ├── incorrect_intervals.html
    ├── effort_impact_delta.html    # Only if test_reasoning was used
    └── ... (up to 19 chart files total)
```

The exported HTML files are self-contained and can be shared without a running server.

---

## Error Handling

The evaluator continues on failures, logging errors and saving partial results:

- API errors are captured in `test_results.json` with `error`, `error_type`, and
  an optional structured `failure_kind`
- Failed generations are counted in `summary.json` under `failed_generations`
- Core generation or MIDI conversion errors are logged but don't halt the evaluation
- The first typed provider rate-limit failure stops only that provider/model's
  queued work; already dispatched work finishes and other model queues continue
- Eval does not automatically retry provider failures or spend additional calls
- Eval-owned logs are written to `<output_dir>/<run-id>/run.log`
- Host application, root logger, and unrelated library output are not redirected

## Performance Notes

- **Cloud providers** use evenly spaced per-model starts plus independent model/global concurrency caps
- **Ollama** runs synchronously, sorted by model to minimize GPU memory swaps
- A live Rich progress table displays during evaluation with per-model pass rates, latency, and cost
- Large evaluations (many models x many prompts x many roots) can take significant time and incur API costs

TPM and RPD metadata is preserved but not scheduled. Automatic retries,
`Retry-After` coordination, account-tier discovery, distributed limit sharing,
and resume execution are also out of scope. Throttle and unstarted counters are
JSON-only; this change does not add Rich progress columns or Dash visualizations.

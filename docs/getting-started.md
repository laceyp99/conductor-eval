# Getting started

This page covers installation, the first evaluation, data locations, and the
safeguards around direct execution.

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

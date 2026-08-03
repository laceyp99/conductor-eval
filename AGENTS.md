# Conductor Eval Agent Guide

## Scope

This repository owns evaluation orchestration, deterministic MIDI checks,
result reporting, and the optional Dash analysis UI. Generation must go through
the public `conductor-core` engine rather than routing providers or persisting
generation artifacts inside Eval.

## Key paths

- `src/conductor_eval/evaluator.py`: evaluation orchestration.
- `src/conductor_eval/checks.py`: deterministic MIDI checks.
- `src/conductor_eval/analysis.py`: optional dashboard and exports.
- `tests/`: evaluator, boundary, and direct-run guard coverage.

## Working rules

- Keep evaluation as a consumer of `LoopGenerationEngine`.
- Do not broaden model matrices, run paid providers, or start broad evaluations without explicit approval.
- Preserve the direct-run confirmation guard for expensive examples.
- Keep Dash, pandas, and Plotly in the dashboard extra.
- Use package-relative or configurable output paths suitable for a standalone checkout.
- Do not commit evaluation outputs, credentials, build artifacts, or planning files.

## Validation

Sync the locked development environment, then run:

```powershell
uv sync --locked --all-extras
uv run --locked --all-extras ruff format --check .
uv run --locked --all-extras ruff check .
uv run --locked --all-extras pytest -q
uv build
```

Use deterministic tests for ordinary validation. Before a commit, inspect
`git status` and the intended diff, keep `uv.lock` synchronized with
`pyproject.toml`, and keep generated evaluations uncommitted.

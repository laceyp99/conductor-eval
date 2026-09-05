# Conductor Eval Agent Guide

Treat this file as good defaults rather than hard rules; the developer's stated
preferences in a session override anything here.

## Scope

This repository owns evaluation orchestration, deterministic MIDI checks,
result reporting, and the optional Dash analysis UI. Generation must go through
the public `conductor-core` engine rather than routing providers or persisting
generation artifacts inside Eval.

## Users and Risk

This project is used by developers and evaluators who compare AI-generated MIDI
loops across models, prompts, roots, scales, and reasoning settings.

Prioritize correctness and reproducibility over convenience. Treat these as
higher-severity problems:

- accidentally making paid provider calls;
- changing evaluation results without making the behavior explicit;
- losing or overwriting saved run artifacts;
- weakening deterministic MIDI checks;

Dashboard presentation issues are important but generally lower severity than
evaluation correctness, data safety, or unexpected provider costs.

## Project Intent

Eval measures how well MIDI loop generators respond to prompts and musical
constraints. Its main capabilities are:

- orchestrating evaluations through the public `LoopGenerationEngine`;
- testing generated MIDI with deterministic checks;
- recording configuration, results, failures, latency, and cost;
- supporting local and cloud model comparisons;
- analyzing saved runs through the optional Dash dashboard.

Keep these responsibilities separate: Core owns generation, Eval owns
evaluation and reporting, and the dashboard consumes saved evaluation data.

## Glossary

- **Evaluation**: A run that generates loops and checks their musical properties.
- **Run**: One saved evaluation, including its configuration, results, and artifacts.
- **Check**: A deterministic validation applied to generated MIDI. Might also be referenced as "musical tests".
- **Dashboard**: The optional Plotly Dash app for exploring saved results.
- **Core**: The `conductor-core` package that generates MIDI loops.
- **Provider**: A model or service route used by Core for generation.

## Key paths

- `src/conductor_eval/evaluator.py`: evaluation orchestration.
- `src/conductor_eval/checks.py`: deterministic MIDI checks.
- `src/conductor_eval/analysis.py`: optional dashboard and exports.
- `tests/`: evaluator, boundary, and direct-run guard coverage.

## Working rules

- Follow “measure twice, cut once” and YAGNI.
- Keep evaluation as a consumer of `LoopGenerationEngine`.
- Do not run paid providers or start broad evaluations without explicit approval.
- Preserve the direct-run confirmation guard for expensive examples.
- Keep Dash, pandas, and Plotly in the dashboard extra.
- Use package-relative or configurable output paths suitable for a standalone checkout.
- Do not commit evaluation outputs, credentials, build artifacts, or planning files.
- Keep relevant documentation and the changelog in sync with behavior changes.

## Validation

See the [README installation section](README.md#installation) for environment
setup and the [README testing section](README.md#configuring-tests) for
evaluation-specific usage.
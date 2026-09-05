# Conductor Eval

Conductor Eval measures how well AI models generate MIDI loops against musical
constraints. It runs evaluations through the public `conductor-core` engine,
applies deterministic MIDI checks, saves results, and provides an optional
Plotly Dash dashboard for analysis.

Use the guides below to get oriented, run an evaluation, inspect its results,
or contribute to the project.

## Guides

- [Getting started](getting-started.md): install Eval, run a first evaluation,
  understand data locations, and review execution safeguards.
- [Evaluation guide](evaluation.md): configure prompts, models, reasoning
  variations, checks, and saved output.
- [Analysis](analysis.md): launch the optional dashboard and explore exported
  results.
- [Development](development.md): run validation, preview the docs, and
  troubleshoot common issues.

## Project boundaries

Core owns generation and provider routing. Eval owns evaluation, deterministic
checks, and reporting. The dashboard reads saved runs and does not generate
loops.

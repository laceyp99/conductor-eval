<div align="center">
  <img src="app/readme-logo.png" alt="Conductor Eval Logo" width="50%">
</div>

**Conductor Eval** measures how well AI models generate MIDI loops against musical
constraints. It runs evaluations through the public [conductor-core](https://github.com/laceyp99/conductor-core) engine,
applies deterministic MIDI checks, saves results, and provides an optional
**Plotly Dash** dashboard for analysis.

## Documentation

The detailed documentation is built with **MkDocs**:

To preview the documentation locally, install the docs dependency group and
start **MkDocs**:

```powershell
uv sync --locked --group docs
uv run --locked --group docs mkdocs serve
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.

**Alternatively**, [Read the documentation source](docs/index.md) as plain markdown.

## Setup

For a short local setup, install the locked development environment and run
the checks:

```powershell
uv sync --locked --all-extras
uv run --locked --all-extras pytest -q
```

See [getting started](docs/getting-started.md) for guidance on running an evaluation.

See [analysis](docs/analysis.md) to run a dashboard displaying evaluation results.


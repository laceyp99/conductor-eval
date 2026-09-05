# Conductor Eval

Conductor Eval measures how well AI models generate MIDI loops against musical
constraints. It runs evaluations through the public `conductor-core` engine,
applies deterministic MIDI checks, saves results, and provides an optional
Plotly Dash dashboard for analysis.

The detailed documentation is built with MkDocs:

- [Read the documentation source](docs/index.md)

For a short local setup, install the locked development environment and run
the checks:

```powershell
uv sync --locked --all-extras
uv run --locked --all-extras pytest -q
```

See the [documentation](docs/index.md) for evaluation examples, output
layouts, dashboard usage, provider safeguards, and the full validation suite.

To preview the documentation locally, install the docs dependency group and
start MkDocs:

```powershell
uv sync --locked --group docs
uv run --locked --group docs mkdocs serve
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser. MkDocs
reloads the local site when documentation files change.

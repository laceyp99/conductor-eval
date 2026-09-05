# Development

## Set up and validate

Use the [installation instructions](getting-started.md#installation) to create the
locked environment. Run the full local validation with:

```powershell
uv sync --locked --all-extras
uv run --locked --all-extras ruff format --check .
uv run --locked --all-extras ruff check .
uv run --locked --all-extras pytest -q
uv build
```

Use focused deterministic tests while iterating. Do not run paid providers or
commit generated evaluations, credentials, build artifacts, or planning files.

## Documentation preview

After installing the documentation tools, preview the site with:

```powershell
uv run --group docs mkdocs serve
```

The documentation opens at [http://127.0.0.1:8000/](http://127.0.0.1:8000/). 

Build it without starting a server:

```powershell
uv run --group docs mkdocs build --strict
```

## Project boundaries

- `src/conductor_eval/evaluator.py`: evaluation orchestration.
- `src/conductor_eval/checks.py`: deterministic MIDI checks.
- `src/conductor_eval/midi.py`: MIDI helpers.
- `src/conductor_eval/paths.py`: data and output paths.
- `src/conductor_eval/analysis.py`: optional dashboard and exports.
- `tests/`: evaluator, checks, paths, boundaries, analysis, and safeguards.

Generation must continue to go through the public `LoopGenerationEngine`.
Provider routing and generation artifacts belong to `conductor-core`.

## Error Handling

The evaluator continues on failures, logging errors and saving partial results:

- API errors are captured in `test_results.json` with an `"error"` field
- Failed generations are counted in `summary.json` under `failed_generations`
- Core generation or MIDI conversion errors are logged but don't halt the evaluation
- Eval-owned logs are written to `<output_dir>/<run-id>/run.log`
- Host application, root logger, and unrelated library output are not redirected

## Performance Notes

- **Cloud providers** run asynchronously with at most four concurrent requests per provider by
  default. 
    - Set `max_cloud_concurrency` on `evaluate()` to adjust this cap. 
    - This is a concurrency guard, not a request-rate limiter, and it does not guarantee compliance with RPM, TPM, or RPD
  quotas. 
    - Reduce
  `max_cloud_concurrency` or split work into smaller evaluations when using lower account limits or
  running expensive workloads. 
    - Persistent provider throttling is recorded as `rate_limited`
  rather than hidden as a generation failure.
- **Ollama** runs synchronously, sorted by model to minimize GPU memory swaps
- A live **Rich** progress table displays during evaluation with per-model pass rates, latency, and cost
- Large evaluations (many models x many prompts x many roots) can take significant time and incur API costs

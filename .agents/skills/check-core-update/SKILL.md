---
name: check-core-update
description: Use when the user asks to check for a new conductor-core version, upgrade the conductor-core dependency, sync conductor-eval with a new Core release, or review Core release notes/changelog. Triggers include "check version of core", "update conductor-core", or "did core release anything new".
---

# Check conductor-core for updates

## Boundary to preserve

`conductor-eval` is a consumer of Core's public `LoopGenerationEngine`. Core owns
generation, provider routing, loop parsing and MIDI conversion, and canonical
generation artifacts. Eval owns evaluation orchestration, deterministic checks,
result reporting, retained run data, and the optional dashboard. Do not copy new
Core implementation into Eval to work around an API change; adapt Eval to the
new public Core API or report that the release is not currently compatible.

The dependency is a Git URL pinned to a release tag in `pyproject.toml`. The
resolved commit is recorded in `uv.lock`; keep both files consistent.

## Workflow

1. Always check first. Read the `conductor-core` dependency in `pyproject.toml`
   and record its tag before considering any file changes.
   Confirm the same tag and resolved commit in `uv.lock`. If they disagree,
   report the existing inconsistency before changing anything.

2. Discover stable tags/releases from `laceyp99/conductor-core` using `gh`
   release/tag commands or `git ls-remote --tags`. Compare versions as semantic
   versions, not as strings, and do not select a prerelease unless the user asks
   for it. If the current pin is the newest applicable tag, report that Eval is
   already current and make no edits.

3. For every intervening release, read the GitHub release notes and the matching
   sections of Core's `CHANGELOG.md`, when available. Also compare the old and
   target tags for the exact public symbols Eval imports; release notes alone are
   not sufficient evidence of compatibility.

4. Inventory current Core usage with a repository search such as
   `rg "conductor_core|LoopGenerationEngine" src tests`. At minimum, assess:

   - `src/conductor_eval/evaluator.py`: `EngineConfig`, `GenerationRequest`,
     `LoopGenerationEngine`, generation-result fields (`midi_path`, `messages`,
     `cost`), `ProviderRateLimitError`, `DURATION_KEYWORDS`, `get_model_info`, and
     the Ollama provider metadata API.
   - `src/conductor_eval/checks.py`: scale, duration, and pitch/interval constants
     and helpers imported from `conductor_core.music`.
   - `src/conductor_eval/analysis.py`: note, duration, and interval display helpers
     used by the optional dashboard.
   - Tests importing Core types or encoding the adapter, artifact, error, model
     metadata, and package-boundary contracts.

   Check signatures, import locations, defaults, return shapes, exceptions, model
   metadata, and artifact lifecycle. Treat changed music constants or semantics as
   potentially evaluation-changing even when imports still succeed. State whether
   each relevant release is drop-in, needs an Eval adaptation, or is blocked.

5. Stop after the check and impact report unless the user instructed you to
   update, upgrade, or sync files. A request to check versions or review release
   notes does not authorize dependency or code changes.

6. When instructed to update, change the tag in `pyproject.toml`, then regenerate
   `uv.lock` with `uv` so it resolves that exact tag and commit. Do not hand-edit
   the lockfile's resolved commit. Make only compatibility changes required by
   the new public API, and update relevant documentation and changelog if behavior
   or recorded evaluation meaning changes.

7. After making changes, validate without making provider calls:

   ```powershell
   uv sync --locked --all-extras
   uv run --locked --all-extras ruff format --check .
   uv run --locked --all-extras ruff check .
   uv run --locked --all-extras pytest -q
   uv build
   uv run --group docs mkdocs build --strict
   ```

   During iteration, prioritize adapter/evaluator, deterministic-check, analysis,
   package-boundary, and direct-run-guard tests. Preserve the exact confirmation
   guard that prevents accidental broad paid evaluations. Do not run Eval's direct
   entry point, live providers, or broad evaluations as migration validation.

## Report

Report the current pin, newest applicable release, inspected intervening
releases, and relevant API or semantic changes. When an update was instructed,
also report files changed, the new lockfile commit, and every validation result.
Explain any breaking change or evaluation-result change explicitly. If blocked,
identify the incompatible Core contract and the smallest upstream or Eval-side
resolution without weakening deterministic checks or the Core/Eval ownership
boundary.

## Safety

- Never make paid or live provider calls without explicit approval.
- Never discard or overwrite saved evaluation runs.
- Do not commit credentials, generated evaluations, Core artifacts, build output,
  or planning files.
- Do not upgrade through a breaking release silently or change checks merely to
  make existing tests pass.

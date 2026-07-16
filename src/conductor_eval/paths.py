"""Resolve Conductor Eval's mutable data directories."""

import os
from pathlib import Path
from typing import Mapping

PROJECT_ID = "eval"
PROJECT_DATA_ENV = "CONDUCTOR_EVAL_HOME"
SUITE_HOME_ENV = "CONDUCTOR_HOME"

# Backward-compatible name for callers using the initial Eval path API.
PROJECT_HOME_ENV = PROJECT_DATA_ENV


def _environment_path(name: str, environ: Mapping[str, str] | None = None) -> Path | None:
    """Return an expanded environment path when the variable is set."""
    env = os.environ if environ is None else environ
    value = env.get(name)
    return Path(value).expanduser() if value else None


def resolve_conductor_home(environ: Mapping[str, str] | None = None) -> Path:
    """Return the shared Conductor suite root without creating it."""
    return _environment_path(SUITE_HOME_ENV, environ) or Path.home() / ".conductor"


def resolve_data_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Return Eval's complete data directory without creating it."""
    return _environment_path(PROJECT_DATA_ENV, environ) or (
        resolve_conductor_home(environ) / PROJECT_ID
    )


def resolve_default_evaluations_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Return Eval's default evaluation-output directory without creating it."""
    return resolve_data_dir(environ) / "evaluations"


def get_data_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Return Eval's data directory without creating it."""
    return resolve_data_dir(environ)


def get_evaluations_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Return the default evaluation-output directory without creating it."""
    return resolve_default_evaluations_dir(environ)

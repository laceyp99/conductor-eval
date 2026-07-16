import importlib
from pathlib import Path

from conductor_eval import paths


def test_default_data_dir_uses_home(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.PROJECT_HOME_ENV, raising=False)
    monkeypatch.delenv(paths.SUITE_HOME_ENV, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))

    assert paths.get_data_dir() == tmp_path / "user-home" / ".conductor" / "eval"


def test_suite_home_override(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.PROJECT_HOME_ENV, raising=False)
    monkeypatch.setenv(paths.SUITE_HOME_ENV, str(tmp_path / "suite-home"))

    assert paths.get_data_dir() == tmp_path / "suite-home" / "eval"


def test_project_home_override_is_complete_directory(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.SUITE_HOME_ENV, raising=False)
    monkeypatch.setenv(paths.PROJECT_HOME_ENV, str(tmp_path / "eval-home"))

    assert paths.get_data_dir() == tmp_path / "eval-home"


def test_project_home_takes_precedence_over_suite_home(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.PROJECT_HOME_ENV, str(tmp_path / "eval-home"))
    monkeypatch.setenv(paths.SUITE_HOME_ENV, str(tmp_path / "suite-home"))

    assert paths.get_data_dir() == tmp_path / "eval-home"


def test_overrides_expand_tilde(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "user-home"))
    monkeypatch.setenv(paths.PROJECT_HOME_ENV, "~/custom-eval")

    assert paths.get_data_dir() == tmp_path / "user-home" / "custom-eval"


def test_import_and_resolution_do_not_create_directories(monkeypatch, tmp_path):
    data_dir = tmp_path / "missing" / "eval-home"
    monkeypatch.setenv(paths.PROJECT_HOME_ENV, str(data_dir))

    importlib.reload(paths)

    assert paths.get_evaluations_dir() == data_dir / "evaluations"
    assert not data_dir.exists()


def test_resolution_is_independent_of_cwd_and_module_location(monkeypatch, tmp_path):
    suite_home = tmp_path / "suite-home"
    fake_site_packages = tmp_path / "venv" / "Lib" / "site-packages" / "conductor_eval"
    cwd = tmp_path / "working-directory"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(paths, "__file__", str(fake_site_packages / "paths.py"))
    monkeypatch.delenv(paths.PROJECT_HOME_ENV, raising=False)
    monkeypatch.setenv(paths.SUITE_HOME_ENV, str(suite_home))

    assert paths.get_data_dir() == suite_home / "eval"


def test_explicit_output_directory_retains_precedence(monkeypatch, tmp_path):
    from conductor_eval import Evaluator

    monkeypatch.setenv(paths.PROJECT_HOME_ENV, str(tmp_path / "eval-home"))
    explicit = tmp_path / "portable-output"

    evaluator = Evaluator(output_dir=explicit)

    assert evaluator.output_dir == explicit

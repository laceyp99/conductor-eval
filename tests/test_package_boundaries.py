from pathlib import Path

import conductor_eval
from conductor_eval import EvalEngineAdapter, Evaluator, get_evaluations_dir


def test_public_package_exports_evaluator_api():
    assert conductor_eval.EvalEngineAdapter is EvalEngineAdapter
    assert conductor_eval.Evaluator is Evaluator


def test_evaluator_default_output_uses_eval_data_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("CONDUCTOR_EVAL_HOME", str(tmp_path / "eval-home"))
    monkeypatch.setattr(Evaluator, "_setup_logging", lambda self: None)

    evaluator = Evaluator()

    assert evaluator.output_dir == get_evaluations_dir()
    assert evaluator.output_dir == tmp_path / "eval-home" / "evaluations"


def test_package_source_has_no_legacy_import_or_path_mutation():
    package_dir = Path(__file__).resolve().parents[1] / "src" / "conductor_eval"
    forbidden_fragments = (
        "sys.path",
        "from src",
        "import src",
    )

    for source_path in package_dir.glob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{fragment!r} found in {source_path}"

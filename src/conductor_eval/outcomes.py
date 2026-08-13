"""Shared result outcome compatibility helpers."""


def get_overall_status(result: dict) -> str:
    """Return the persisted outcome, with fallbacks for legacy results."""
    tests = result.get("tests", {})
    if "overall_status" in tests:
        return tests["overall_status"]
    if result.get("error"):
        return "generation_error"
    if tests.get("overall_pass", False):
        return "passed"
    return "failed"

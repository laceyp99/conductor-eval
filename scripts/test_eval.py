"""Run a small, intentional evaluation against local Ollama models.

Use this script as a starting point when experimenting with Eval settings. It
keeps the model list explicit so running it cannot silently expand into the
paid cloud-model matrix supported by :class:`conductor_eval.Evaluator`.

Before running it:
    1. Install Conductor Eval and its dependencies.
    2. Start Ollama and make sure the models below are installed.
    3. Run ``.\\.venv\\Scripts\\python.exe scripts\test_eval.py`` from the repo root.

Results are written beneath Eval's default
``~/.conductor/eval/evaluations/`` directory using the evaluator's timestamped
run-directory layout. ``CONDUCTOR_EVAL_HOME`` and ``CONDUCTOR_HOME`` can
override that location.
"""

from conductor_eval import Evaluator


def main() -> None:
    """Run an intentional local-model evaluation using Eval's default output."""

    # --- Evaluation-wide settings -----------------------------------------
    # A temperature of zero favors repeatable output. Provider responses can
    # still vary, so this does not make a model evaluation fully deterministic.
    evaluator = Evaluator(temperature=0.0)

    # --- Evaluation matrix ------------------------------------------------
    # Eval appends "in {root} {scale}" to the prompt and tests both major and
    # minor scales. With 1 prompt, 2 roots, and 2 models, this produces eight
    # generation tasks: 1 prompt x 2 roots x 2 scales x 2 models.
    evaluator.evaluate(
        # Duration checks recognize "quarter notes" in this prompt.
        prompts="an arpeggiator using only quarter notes",
        # Add roots carefully: each one multiplies the number of tasks by the
        # two scales that Eval always checks.
        roots=["C", "G"],
        # Explicit local model names prevent an accidental broad or paid run.
        # Replace these values with model names installed in your Ollama setup.
        models=["llama3.2:1b", "granite4.1:3b"],
        # The run name becomes part of evaluations/<timestamp>_<run_name>/
        # beneath Eval's resolved data directory.
        run_name="test_eval",
    )


if __name__ == "__main__":
    # This guard keeps importing the script from immediately starting a run.
    main()

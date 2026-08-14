"""
Evaluator class for unified MIDI loop generation testing across models.

This module provides a flexible evaluation framework that can:
- Test multiple prompts across multiple models
- Auto-detect test parameters from prompt text
- Support async execution for cloud providers and sync for local (Ollama)
- Save structured results including MIDI files, chat history, and test results
"""

import asyncio
import hashlib
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Union
from uuid import uuid4

from conductor_core import EngineConfig, GenerationRequest, LoopGenerationEngine
from conductor_core.errors import ProviderRateLimitError
from conductor_core.music import DURATION_KEYWORDS, get_model_info
from conductor_core.providers import ollama as ollama_api
from mido import MidiFile
from rich.console import Console
from rich.live import Live
from rich.table import Table

from conductor_eval.checks import (
    chord_event_positions_test,
    chord_progression_test,
    duration_test,
    harmonic_rhythm_test,
    monophony_test,
    polyphony_test,
    scale_test,
)
from conductor_eval.outcomes import get_overall_status
from conductor_eval.paths import get_evaluations_dir

DIRECT_EVALUATION_CONFIRMATION = "RUN CLOUD EVALUATION"


class EvalEngineAdapter:
    """Translate evaluation tasks into Conductor Core generation requests.

    Core owns provider routing, loop parsing, MIDI conversion, and generation
    artifact persistence. Eval only loads the Core-produced MIDI so it can run
    checks and copy the result into the evaluation report layout.
    """

    def __init__(
        self,
        artifact_root: str | Path,
        engine: LoopGenerationEngine | None = None,
    ):
        self.artifact_root = Path(artifact_root)
        self.engine = engine or LoopGenerationEngine(
            config=EngineConfig.from_defaults(artifact_root=self.artifact_root)
        )

    def generate(
        self,
        *,
        description: str,
        key: str,
        scale: str,
        model: str,
        temperature: float,
        use_thinking: bool,
        effort: str | None,
    ) -> tuple[MidiFile, list[dict], float | None]:
        core_result = self.engine.generate(
            GenerationRequest(
                key=key,
                scale=scale,
                description=description,
                model=model,
                temperature=temperature,
                use_thinking=use_thinking,
                effort=effort or "low",
                render_audio=False,
            )
        )
        return MidiFile(core_result.midi_path), core_result.messages, core_result.cost


def confirm_direct_evaluation(input_func=None) -> bool:
    """
    Confirm before running the expensive direct-execution evaluation.

    Returns:
        bool: True when the exact confirmation phrase is entered.
    """
    if input_func is None:
        input_func = input

    print(
        "WARNING: This direct evaluator run starts a broad cloud evaluation "
        "across multiple paid providers."
    )
    print("It may be slow and may incur API costs.")
    try:
        response = input_func(f"Type {DIRECT_EVALUATION_CONFIRMATION!r} to continue: ")
    except EOFError:
        response = ""

    if response != DIRECT_EVALUATION_CONFIRMATION:
        print("Aborted. No evaluator was created and no provider calls were made.")
        return False

    return True


class Evaluator:
    """
    Unified evaluation framework for testing MIDI loop generation across models.

    Usage:
        evaluator = Evaluator()
        results = evaluator.evaluate(
            prompts="an arpeggiator using only quarter notes",
            roots=["C", "G"],
            models="openai",
            run_name="quarter_arp_test"
        )

    Attributes:
        SCALES: List of scales to test (always major and minor)
        AVAILABLE_TESTS: Registry of available test functions
    """

    SCALES = ["major", "minor"]
    MAX_CLOUD_CONCURRENCY = 4

    AVAILABLE_TESTS = {
        "scale": scale_test,
        "duration": duration_test,
        "monophony": monophony_test,
        "polyphony": polyphony_test,
        "chord_progression": chord_progression_test,
        "harmonic_rhythm": harmonic_rhythm_test,
        "chord_event_positions": chord_event_positions_test,
    }

    @classmethod
    def _with_required_scale_test(cls, tests: list[str]) -> list[str]:
        """Validate selected tests and include the always-on scale check once."""
        unknown = sorted(set(tests) - set(cls.AVAILABLE_TESTS))
        if unknown:
            raise ValueError("Unknown tests: " + ", ".join(unknown))
        if "scale" in tests:
            return list(tests)
        return ["scale", *tests]

    @staticmethod
    def _has_substantive_evidence(test_name: str, test_result: dict) -> bool:
        """Return whether a completed check examined enough evidence to be eligible."""
        if test_name in {"scale", "duration"}:
            return test_result.get("total", 0) > 0
        if test_name in {"monophony", "polyphony"}:
            return test_result.get("total_notes", 0) > 0
        if test_name == "chord_progression":
            return bool(test_result.get("bars"))
        if test_name == "harmonic_rhythm":
            return bool(test_result.get("expected_onsets"))
        if test_name == "chord_event_positions":
            return bool(test_result.get("expected_positions"))
        return False

    @staticmethod
    def _is_overall_eligible(test_results: dict) -> bool:
        """Return whether a result belongs in an overall pass-rate denominator."""
        if "overall_status" in test_results:
            return test_results["overall_status"] in {"passed", "failed"}
        return "overall_pass" in test_results

    def __init__(
        self,
        output_dir: str | Path | None = None,
        temperature: float = 0.0,
    ):
        """
        Initialize the Evaluator.

        Args:
            output_dir: Base directory for all evaluation outputs. Defaults to
                the ``evaluations`` subdirectory in Eval's data directory.
            temperature: Default temperature for generation.
        """
        self.output_dir = get_evaluations_dir() if output_dir is None else Path(output_dir)
        self.temperature = temperature
        self.console = Console(force_terminal=True)
        self.model_info = get_model_info()

    @staticmethod
    def _create_run_logger(run_path: Path, run_id: str) -> tuple[logging.Logger, logging.Handler]:
        """Create an isolated file logger for one evaluation run."""
        logger = logging.Logger(f"{__name__}.run.{run_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        file_handler = logging.FileHandler(run_path / "run.log", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(file_handler)
        return logger, file_handler

    @staticmethod
    def _close_run_logger(logger: logging.Logger, handler: logging.Handler) -> None:
        """Flush and release the file handler owned by one evaluation run."""
        handler.flush()
        logger.removeHandler(handler)
        handler.close()

    @staticmethod
    def _format_traceback(error: Exception) -> str:
        """Return traceback locations without exception text or source content."""
        return "\n".join(
            f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}'
            for frame in traceback.extract_tb(error.__traceback__)
        )

    def evaluate(
        self,
        prompts: Union[str, list[str]],
        roots: list[str],
        models: Union[str, list[str]] = "all",
        run_name: str = None,
        tests: list[str] = ["scale", "duration"],
        test_reasoning: bool = False,
        test_params: dict[str, dict] | None = None,
        max_cloud_concurrency: int = MAX_CLOUD_CONCURRENCY,
    ) -> dict:
        """
        Run evaluation across all specified combinations.

        Args:
            prompts: Complete prompt(s) - will have " in {root} {scale}" appended.
            roots: List of root notes to test (e.g., ["C", "D", "F#"]).
            models: "all" | provider name ("openai", "ollama", etc.) | list of model names.
            run_name: Name for this evaluation run (used in output directory). Required.
            tests: List of test names to run (default: ["scale", "duration"]).
                   "scale" always runs. Others auto-detect params from prompt.
            test_reasoning: If True, test all thinking modes and effort levels for compatible models.
            test_params: Explicit keyword arguments for named tests. Duration parameters override
                         prompt detection; omitted duration parameters still use keyword detection.
            max_cloud_concurrency: Maximum simultaneous requests to each cloud provider.

        Returns:
            dict: Summary of evaluation results.

        Raises:
            ValueError: If run_name is not provided.
        """
        if run_name is None:
            raise ValueError("run_name is required")
        if (
            isinstance(max_cloud_concurrency, bool)
            or not isinstance(max_cloud_concurrency, int)
            or max_cloud_concurrency <= 0
        ):
            raise ValueError("max_cloud_concurrency must be a positive integer")

        tests = self._with_required_scale_test(tests)
        test_params = self._validate_test_params(tests, test_params)

        # Normalize prompts to list
        if isinstance(prompts, str):
            prompts = [prompts]

        run_path, run_id, timestamp = self._create_run_directory(run_name)
        logger, handler = self._create_run_logger(run_path, run_id)

        try:
            # Resolve models to (provider, model) tuples.
            resolved_models = self._resolve_models(models, logger)

            # Save configuration
            config = {
                "run_id": run_id,
                "run_name": run_name,
                "timestamp": timestamp,
                "prompts": prompts,
                "roots": roots,
                "scales": self.SCALES,
                "models": [(p, m) for p, m in resolved_models],
                "tests": tests,
                "test_params": test_params,
                "test_reasoning": test_reasoning,
                "temperature": self.temperature,
                "max_cloud_concurrency": max_cloud_concurrency,
            }
            with open(run_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            # Generate all task combinations
            tasks = self._generate_tasks(
                prompts=prompts,
                roots=roots,
                resolved_models=resolved_models,
                tests=tests,
                test_reasoning=test_reasoning,
                test_params=test_params,
            )
            logger.info("Starting evaluation '%s' with %d total tasks", run_name, len(tasks))

            # Separate async and sync tasks
            async_tasks = [t for t in tasks if self._is_async_provider(t["provider"])]
            sync_tasks = [t for t in tasks if not self._is_async_provider(t["provider"])]

            all_results = []

            # Run async tasks (cloud providers)
            if async_tasks:
                async_results = asyncio.run(
                    self._run_async_batch(
                        async_tasks,
                        run_path,
                        tests,
                        logger,
                        max_cloud_concurrency,
                    )
                )
                all_results.extend(async_results)

            # Run sync tasks (Ollama)
            if sync_tasks:
                sync_results = self._run_sync_batch(sync_tasks, run_path, tests, logger)
                all_results.extend(sync_results)

            # Generate and save summary
            summary = self._generate_summary(all_results, config)
            with open(run_path / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            logger.info("Evaluation complete. Results saved to %s", run_path)
            return summary
        except Exception as error:
            logger.error(
                "Evaluation failed: run_path=%s error_type=%s\nTraceback:\n%s",
                run_path,
                type(error).__name__,
                self._format_traceback(error),
            )
            raise
        finally:
            self._close_run_logger(logger, handler)

    def run_tests(
        self,
        midi_data: MidiFile,
        root: str,
        scale: str,
        prompt: str,
        tests: list[str],
        test_params: dict[str, dict] | None = None,
    ) -> dict:
        """
        Run specified tests on MIDI data.

        Args:
            midi_data: The MIDI file object to test.
            root: Root note used in generation.
            scale: Scale used in generation.
            prompt: Original prompt (for parameter detection).
            tests: List of test names to run.
            test_params: Explicit keyword arguments keyed by test name.

        Returns:
            dict: Test results with format:
                {
                    "scale": {...results...},
                    "duration": {...results...} or {"skipped": "reason"},
                    "overall_pass": bool
                }
        """
        tests = self._with_required_scale_test(tests)
        results = {}
        all_passed = True
        substantive_checks = 0
        check_error_occurred = False
        test_params = self._validate_test_params(tests, test_params)

        for test_name in tests:
            if test_name not in self.AVAILABLE_TESTS:
                results[test_name] = {"skipped": f"Unknown test: {test_name}"}
                continue

            test_func = self.AVAILABLE_TESTS[test_name]

            if test_name == "scale":
                # Scale test always runs with provided root/scale
                try:
                    test_result = test_func(midi_data, root, scale)
                    test_result["ran"] = True
                    test_result["eligible"] = self._has_substantive_evidence(test_name, test_result)
                    test_result["passed"] = (
                        test_result["eligible"] and test_result.get("incorrect", 0) == 0
                    )
                    test_result["status"] = (
                        "passed"
                        if test_result["passed"]
                        else "failed"
                        if test_result["eligible"]
                        else "ineligible"
                    )
                    test_result["params"] = {"root": root, "scale": scale}
                    results[test_name] = test_result
                    if test_result["eligible"]:
                        substantive_checks += 1
                    if not test_result["passed"] and test_result["eligible"]:
                        all_passed = False
                except Exception as e:
                    results[test_name] = {
                        "ran": False,
                        "eligible": False,
                        "status": "check_error",
                        "error": str(e),
                    }
                    all_passed = False
                    check_error_occurred = True

            elif test_name == "duration":
                explicit_params = test_params.get(test_name, {})
                detected_params = self._detect_test_params(prompt, test_name)
                resolved_params = {**detected_params, **explicit_params}
                if "duration" not in resolved_params:
                    results[test_name] = {
                        "ran": False,
                        "eligible": False,
                        "status": "ineligible",
                        "skipped": "No duration keyword detected in prompt",
                    }
                else:
                    try:
                        duration_value = resolved_params["duration"]
                        test_result = test_func(midi_data, duration_value)
                        test_result["ran"] = True
                        test_result["eligible"] = self._has_substantive_evidence(
                            test_name, test_result
                        )
                        test_result["passed"] = (
                            test_result["eligible"] and test_result.get("incorrect", 0) == 0
                        )
                        test_result["status"] = (
                            "passed"
                            if test_result["passed"]
                            else "failed"
                            if test_result["eligible"]
                            else "ineligible"
                        )
                        test_result["params"] = {"duration": duration_value}
                        test_result["detected_from_prompt"] = "duration" not in explicit_params
                        results[test_name] = test_result
                        if test_result["eligible"]:
                            substantive_checks += 1
                        if not test_result["passed"] and test_result["eligible"]:
                            all_passed = False
                    except Exception as e:
                        results[test_name] = {
                            "ran": False,
                            "eligible": False,
                            "status": "check_error",
                            "error": str(e),
                        }
                        all_passed = False
                        check_error_occurred = True

            else:
                detected_params = self._detect_test_params(prompt, test_name)
                resolved_params = {**detected_params, **test_params.get(test_name, {})}
                if test_name == "chord_progression":
                    resolved_params.update({"root": root, "scale": scale})
                try:
                    test_result = test_func(midi_data, **resolved_params)
                    test_result["ran"] = True
                    test_result["eligible"] = self._has_substantive_evidence(test_name, test_result)
                    test_result["passed"] = test_result["eligible"] and test_result.get(
                        "passed", test_result.get("incorrect", 0) == 0
                    )
                    test_result["status"] = (
                        "passed"
                        if test_result["passed"]
                        else "failed"
                        if test_result["eligible"]
                        else "ineligible"
                    )
                    test_result["params"] = resolved_params
                    results[test_name] = test_result
                    if test_result["eligible"]:
                        substantive_checks += 1
                    if test_result["eligible"] and not test_result["passed"]:
                        all_passed = False
                except Exception as e:
                    results[test_name] = {
                        "ran": False,
                        "eligible": False,
                        "status": "check_error",
                        "error": str(e),
                    }
                    all_passed = False
                    check_error_occurred = True

        results["overall_pass"] = all_passed and substantive_checks > 0
        results["overall_status"] = (
            "check_error"
            if check_error_occurred
            else "passed"
            if results["overall_pass"]
            else "failed"
            if substantive_checks > 0 and not all_passed
            else "ineligible"
        )
        return results

    @staticmethod
    def _validate_test_params(tests: list[str], test_params: dict[str, dict] | None) -> dict:
        """Validate and copy explicit test arguments."""
        if test_params is None:
            return {}
        if not isinstance(test_params, dict):
            raise ValueError("test_params must be a dictionary keyed by test name")

        unselected = sorted(set(test_params) - set(tests))
        if unselected:
            raise ValueError("test_params contains unselected tests: " + ", ".join(unselected))

        validated = {}
        for test_name, params in test_params.items():
            if not isinstance(params, dict):
                raise ValueError(f"test_params[{test_name!r}] must be a dictionary")
            validated[test_name] = dict(params)
        return validated

    def _resolve_models(
        self, models: Union[str, list[str]], logger: logging.Logger
    ) -> list[tuple[str, str]]:
        """
        Resolve model specification to (provider, model_name) tuples.

        Args:
            models: "all" | provider name | list of model names

        Returns:
            list: List of (provider, model_name) tuples
        """
        resolved = []

        if isinstance(models, str):
            models_lower = models.lower()
            if models_lower == "all":
                # All cloud models from model_list.json
                for provider in ["OpenAI", "Anthropic", "Google"]:
                    if provider in self.model_info["models"]:
                        for model in self.model_info["models"][provider].keys():
                            resolved.append((provider, model))
                # All Ollama models
                resolved.extend(("Ollama", model) for model in self._discover_ollama_models())

            elif models_lower == "openai":
                for model in self.model_info["models"]["OpenAI"].keys():
                    resolved.append(("OpenAI", model))

            elif models_lower == "anthropic":
                for model in self.model_info["models"]["Anthropic"].keys():
                    resolved.append(("Anthropic", model))

            elif models_lower == "google":
                for model in self.model_info["models"]["Google"].keys():
                    resolved.append(("Google", model))

            elif models_lower == "ollama":
                resolved.extend(("Ollama", model) for model in self._discover_ollama_models())

            else:
                # Assume it's a single model name
                provider = self._get_provider(models)
                if provider:
                    resolved.append((provider, models))
                else:
                    raise ValueError(f"Unknown model or provider: {models}")

        elif isinstance(models, list):
            for model in models:
                provider = self._get_provider(model)
                if provider:
                    resolved.append((provider, model))

        return resolved

    @staticmethod
    def _discover_ollama_models() -> list[str]:
        """Return locally available Ollama models, or none when discovery is unavailable."""
        try:
            return list(ollama_api.get_model_list())
        except Exception:
            return []

    def _get_provider(self, model: str) -> str:
        """
        Determine provider for a given model name.

        Args:
            model: Model name string

        Returns:
            str: Provider name or None if not found
        """
        # Check cloud providers first
        for provider in ["OpenAI", "Anthropic", "Google"]:
            if provider in self.model_info["models"]:
                if model in self.model_info["models"][provider]:
                    return provider

        if model in self._discover_ollama_models():
            return "Ollama"

        return None

    def _is_async_provider(self, provider: str) -> bool:
        """
        Return True if provider should use async execution.

        Args:
            provider: Provider name

        Returns:
            bool: True for cloud providers, False for Ollama
        """
        return provider in ["OpenAI", "Anthropic", "Google"]

    def _get_model_capabilities(self, provider: str, model: str) -> dict:
        """
        Get thinking/effort capabilities for a model.

        Args:
            provider: Provider name
            model: Model name

        Returns:
            dict: Capabilities including extended_thinking, etc.
        """
        if provider == "Ollama":
            return {"extended_thinking": False, "effort_options": []}

        if provider in self.model_info["models"]:
            if model in self.model_info["models"][provider]:
                return self.model_info["models"][provider][model]

        return {"extended_thinking": False, "effort_options": []}

    def _detect_test_params(self, prompt: str, test_name: str) -> dict:
        """
        Extract test parameters from prompt text using keyword matching.

        Args:
            prompt: The full prompt string
            test_name: Name of the test

        Returns:
            dict: Detected parameters, or empty dict if not found
        """
        if test_name == "duration":
            prompt_lower = prompt.lower()
            for keyword, duration_value in DURATION_KEYWORDS.items():
                if keyword in prompt_lower:
                    return {"duration": duration_value}
            return {}

        return {}

    def _generate_tasks(
        self,
        prompts: list[str],
        roots: list[str],
        resolved_models: list[tuple[str, str]],
        tests: list[str],
        test_reasoning: bool,
        test_params: dict[str, dict] | None = None,
    ) -> list[dict]:
        """
        Generate all task combinations to run.

        Args:
            prompts: List of base prompts
            roots: List of root notes
            resolved_models: List of (provider, model) tuples
            tests: List of test names
            test_reasoning: Whether to test reasoning variations
            test_params: Explicit keyword arguments keyed by test name

        Returns:
            list: List of task dictionaries
        """
        tasks = []

        for prompt in prompts:
            for root in roots:
                for scale in self.SCALES:
                    full_prompt = f"{prompt} in {root} {scale}"

                    for provider, model in resolved_models:
                        variations = self._generate_variations(
                            model=model,
                            provider=provider,
                            test_reasoning=test_reasoning,
                        )

                        for variation in variations:
                            tasks.append(
                                {
                                    "provider": provider,
                                    "model": model,
                                    "original_prompt": prompt,
                                    "full_prompt": full_prompt,
                                    "root": root,
                                    "scale": scale,
                                    "use_thinking": variation["use_thinking"],
                                    "effort": variation["effort"],
                                    "variation_name": variation["name"],
                                    "test_params": {
                                        name: dict(params)
                                        for name, params in (test_params or {}).items()
                                    },
                                }
                            )

        occurrences: dict[str, int] = {}
        for task in tasks:
            fingerprint = self._task_fingerprint(task)
            occurrence = occurrences.get(fingerprint, 0) + 1
            occurrences[fingerprint] = occurrence
            task["task_id"] = (
                f"task-{self._sanitize_filename(task['original_prompt'], max_len=32)}-"
                f"{fingerprint[:16]}-{occurrence}"
            )

        return tasks

    def _generate_variations(self, model: str, provider: str, test_reasoning: bool) -> list[dict]:
        """
        Generate all config variations to test for a model.

        Args:
            model: Model name
            provider: Provider name
            test_reasoning: Whether to test reasoning variations

        Returns:
            list: List of variation config dictionaries
        """
        variations = []
        capabilities = self._get_model_capabilities(provider, model)
        supports_thinking = capabilities.get("extended_thinking", False)
        effort_options = capabilities.get("effort_options", [])

        if test_reasoning and supports_thinking:
            # For OpenAI reasoning models (o-series), only effort levels matter.
            if provider == "OpenAI" and supports_thinking:
                for effort in effort_options:
                    variations.append(
                        {
                            "use_thinking": True,
                            "effort": effort,
                            "name": effort,
                        }
                    )
            # For Anthropic/Google, test thinking with effort levels when supported.
            elif provider in ["Anthropic", "Google"] and effort_options:
                for effort in effort_options:
                    variations.append(
                        {
                            "use_thinking": True,
                            "effort": effort,
                            "name": effort,
                        }
                    )
            # For Anthropic/Google with a reasoning toggle but no effort options.
            elif provider in ["Anthropic", "Google"]:
                variations.append(
                    {
                        "use_thinking": False,
                        "effort": None,
                        "name": "standard",
                    }
                )
                variations.append(
                    {
                        "use_thinking": True,
                        "effort": None,
                        "name": "w_reasoning",
                    }
                )
            else:
                variations.append(
                    {
                        "use_thinking": False,
                        "effort": None,
                        "name": "standard",
                    }
                )
        else:
            # No reasoning testing: use the default effort for effort-based models.
            if supports_thinking and provider == "OpenAI":
                variations.append(
                    {
                        "use_thinking": True,
                        "effort": effort_options[0],
                        "name": effort_options[0],
                    }
                )
            elif effort_options and provider in ["Anthropic", "Google"]:
                variations.append(
                    {
                        "use_thinking": True,
                        "effort": effort_options[0],
                        "name": effort_options[0],
                    }
                )
            else:
                variations.append(
                    {
                        "use_thinking": False,
                        "effort": None,
                        "name": "standard",
                    }
                )

        return variations

    async def _run_async_batch(
        self,
        tasks: list[dict],
        run_path: Path,
        tests_to_run: list[str],
        logger: logging.Logger,
        max_cloud_concurrency: int = MAX_CLOUD_CONCURRENCY,
    ) -> list[dict]:
        """
        Run cloud tasks with an independent concurrency cap for each provider.

        Args:
            tasks: List of task dictionaries
            run_path: Path to save results
            tests_to_run: List of test names to run

        Returns:
            list: List of result dictionaries
        """
        providers = {task["provider"] for task in tasks}
        semaphores = {provider: asyncio.Semaphore(max_cloud_concurrency) for provider in providers}

        results = []
        total_tasks = len(tasks)

        # Create live table for progress
        table = Table(title="Evaluation Progress")
        table.add_column("Provider")
        table.add_column("Model")
        table.add_column("Variation")
        table.add_column("Progress")
        table.add_column("Pass Rate")
        table.add_column("Avg Latency")
        table.add_column("Avg Cost")

        async def run_single_task(task: dict) -> dict:
            provider = task["provider"]
            async with semaphores[provider]:
                return await asyncio.to_thread(
                    self._run_single,
                    task=task,
                    run_path=run_path,
                    tests_to_run=tests_to_run,
                    logger=logger,
                )

        with Live(table, console=self.console, refresh_per_second=2) as live:
            # Create all async tasks
            async_tasks = [run_single_task(t) for t in tasks]

            # Run with gathering and update progress
            for coro in asyncio.as_completed(async_tasks):
                result = await coro
                results.append(result)

                # Update table
                new_table = Table(title=f"Evaluation Progress ({len(results)}/{total_tasks})")
                new_table.add_column("Provider")
                new_table.add_column("Model")
                new_table.add_column("Eligible")
                new_table.add_column("Pass Rate")
                new_table.add_column("Avg Latency")
                new_table.add_column("Avg Cost")

                # Aggregate stats by model
                stats = {}
                for r in results:
                    key = (r["provider"], r["model"])
                    s = stats.setdefault(
                        key,
                        {
                            "eligible": 0,
                            "passed": 0,
                            "latency_sum": 0.0,
                            "latency_count": 0,
                            "cost_sum": 0.0,
                            "cost_count": 0,
                        },
                    )
                    test_results = r.get("tests", {})
                    if self._is_overall_eligible(test_results):
                        s["eligible"] += 1
                    if test_results.get("overall_pass", False):
                        s["passed"] += 1
                    latency = r.get("metrics", {}).get("api_latency")
                    if latency is not None:
                        s["latency_sum"] += latency
                        s["latency_count"] += 1
                    cost = r.get("metrics", {}).get("cost")
                    if cost is not None:
                        s["cost_sum"] += cost
                        s["cost_count"] += 1

                for (provider, model), s in stats.items():
                    pass_rate = s["passed"] / s["eligible"] * 100 if s["eligible"] else 0
                    avg_latency = (
                        s["latency_sum"] / s["latency_count"] if s["latency_count"] else None
                    )
                    avg_cost = s["cost_sum"] / s["cost_count"] if s["cost_count"] else None
                    new_table.add_row(
                        provider,
                        model,
                        str(s["eligible"]),
                        f"{pass_rate:.1f}%",
                        f"{avg_latency:.2f}s" if avg_latency is not None else "N/A",
                        f"${avg_cost:.4f}" if avg_cost is not None else "N/A",
                    )

                live.update(new_table)

        return results

    def _run_sync_batch(
        self,
        tasks: list[dict],
        run_path: Path,
        tests_to_run: list[str],
        logger: logging.Logger,
    ) -> list[dict]:
        """
        Run tasks synchronously (for Ollama).

        Args:
            tasks: List of task dictionaries
            run_path: Path to save results
            tests_to_run: List of test names to run

        Returns:
            list: List of result dictionaries
        """
        # Sort tasks by model to minimize GPU memory swaps
        tasks = sorted(tasks, key=lambda t: t["model"])

        results = []
        total_tasks = len(tasks)

        table = Table(title="Ollama Evaluation Progress")
        table.add_column("Model")
        table.add_column("Progress")
        table.add_column("Pass Rate")
        table.add_column("Avg Latency")

        with Live(table, console=self.console, refresh_per_second=2) as live:
            for i, task in enumerate(tasks):
                result = self._run_single(task, run_path, tests_to_run, logger)
                results.append(result)

                # Update table
                new_table = Table(title=f"Evaluation Progress ({len(results)}/{total_tasks})")
                new_table.add_column("Model")
                new_table.add_column("Eligible")
                new_table.add_column("Pass Rate")
                new_table.add_column("Avg Latency")

                # Aggregate stats by model
                stats = {}
                for r in results:
                    key = r["model"]
                    s = stats.setdefault(
                        key,
                        {
                            "eligible": 0,
                            "passed": 0,
                            "latency_sum": 0.0,
                            "latency_count": 0,
                        },
                    )
                    test_results = r.get("tests", {})
                    if self._is_overall_eligible(test_results):
                        s["eligible"] += 1
                    if test_results.get("overall_pass", False):
                        s["passed"] += 1
                    latency = r.get("metrics", {}).get("api_latency")
                    if latency is not None:
                        s["latency_sum"] += latency
                        s["latency_count"] += 1

                for model, s in stats.items():
                    pass_rate = s["passed"] / s["eligible"] * 100 if s["eligible"] else 0
                    avg_latency = (
                        s["latency_sum"] / s["latency_count"] if s["latency_count"] else None
                    )
                    new_table.add_row(
                        model,
                        str(s["eligible"]),
                        f"{pass_rate:.1f}%",
                        f"{avg_latency:.2f}s" if avg_latency is not None else "N/A",
                    )

                live.update(new_table)

        return results

    def _run_single(
        self,
        task: dict,
        run_path: Path,
        tests_to_run: list[str],
        logger: logging.Logger | None = None,
    ) -> dict:
        """
        Run single generation, tests, and save results.

        Args:
            task: Task dictionary with all parameters
            run_path: Path to save results
            tests_to_run: List of test names to run

        Returns:
            dict: Result dictionary
        """
        provider = task["provider"]
        model = task["model"]
        full_prompt = task["full_prompt"]
        original_prompt = task["original_prompt"]
        root = task["root"]
        scale = task["scale"]
        use_thinking = task["use_thinking"]
        effort = task["effort"]

        # Build result structure
        result = {
            "task_id": task.get("task_id"),
            "model": model,
            "provider": provider,
            "prompt": full_prompt,
            "original_prompt": original_prompt,
            "root": root,
            "scale": scale,
            "config": {
                "use_thinking": use_thinking,
                "effort": effort,
                "temperature": self.temperature,
                "variation_name": task["variation_name"],
            },
            "metrics": {
                "api_latency": None,
                "attempt_latency": None,
                "cost": None,
                "cost_available": False,
            },
            "tests": {},
            "error": None,
        }
        logger = logger or logging.Logger(__name__)

        # Generate MIDI
        start_time = time.perf_counter()
        try:
            adapter = EvalEngineAdapter(run_path / "core_artifacts")
            midi_file, messages, cost = adapter.generate(
                description=original_prompt,
                key=root,
                scale=scale,
                model=model,
                temperature=self.temperature,
                use_thinking=use_thinking,
                effort=effort,
            )
            time_elapsed = time.perf_counter() - start_time

            result["metrics"]["api_latency"] = time_elapsed
            result["metrics"]["attempt_latency"] = time_elapsed
            result["metrics"]["cost"] = cost
            result["metrics"]["cost_available"] = cost is not None
        except Exception as e:
            result["metrics"]["attempt_latency"] = time.perf_counter() - start_time
            logger.error(
                "Task failed: task_id=%s provider=%s model=%s root=%s scale=%s variation=%s "
                "error_type=%s\nTraceback:\n%s",
                task.get("task_id"),
                provider,
                model,
                root,
                scale,
                task["variation_name"],
                type(e).__name__,
                self._format_traceback(e),
            )
            result["error"] = str(e)
            result["tests"]["overall_pass"] = False
            result["tests"]["overall_status"] = (
                "rate_limited" if isinstance(e, ProviderRateLimitError) else "generation_error"
            )
            # Still save the result even on failure
            self._save_results(result, None, [], run_path, task)
            return result

        # Run tests
        test_results = self.run_tests(
            midi_data=midi_file,
            root=root,
            scale=scale,
            prompt=original_prompt,
            tests=tests_to_run,
            test_params=task.get("test_params"),
        )
        result["tests"] = test_results
        # Save results
        self._save_results(result, midi_file, messages, run_path, task)

        return result

    def _save_results(
        self,
        result: dict,
        midi_data: MidiFile,
        messages: list,
        run_path: Path,
        task: dict,
    ) -> None:
        """
        Save MIDI, messages, and test results to disk.

        Args:
            result: Result dictionary
            midi_data: MIDI file object (or None on failure)
            messages: Chat messages list
            run_path: Base path for this run
            task: Task dictionary with path info
        """
        task_id = task["task_id"]
        result_dir = run_path / "results" / task_id
        result_dir.mkdir(parents=True, exist_ok=False)

        # Save MIDI
        if midi_data is not None:
            midi_path = result_dir / "loop.mid"
            midi_data.save(str(midi_path))

        # Save messages (for fine-tuning)
        messages_path = result_dir / "messages.json"
        with open(messages_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)

        # Save test results
        results_path = result_dir / "test_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    def _generate_summary(self, all_results: list[dict], config: dict) -> dict:
        """
        Aggregate results into summary statistics.

        Args:
            all_results: List of all result dictionaries
            config: Original configuration dictionary

        Returns:
            dict: Summary with aggregated statistics
        """
        summary = {
            "run_id": config["run_id"],
            "config": config,
            "totals": {
                "total_generations": len(all_results),
                "successful_generations": 0,
                "failed_generations": 0,
                "generation_error_generations": 0,
                "rate_limited_generations": 0,
                "check_error_generations": 0,
                "validation_failed_generations": 0,
                "ineligible_generations": 0,
                "eligible_generations": 0,
                "overall_pass_count": 0,
                "overall_pass_rate": 0.0,
                "total_cost": 0.0,
                "known_cost_generations": 0,
                "unknown_cost_generations": 0,
                "total_time": 0.0,
                "successful_latency_total": 0.0,
                "successful_latency_count": 0,
                "avg_successful_latency": None,
            },
            "by_model": {},
            "by_root": {},
            "by_scale": {},
        }

        for r in all_results:
            outcome = get_overall_status(r)

            # Totals
            if r.get("error"):
                summary["totals"]["failed_generations"] += 1
                if outcome == "rate_limited":
                    summary["totals"]["rate_limited_generations"] += 1
                else:
                    summary["totals"]["generation_error_generations"] += 1
            else:
                summary["totals"]["successful_generations"] += 1

            if outcome in {"passed", "failed"}:
                summary["totals"]["eligible_generations"] += 1
            if outcome == "failed":
                summary["totals"]["validation_failed_generations"] += 1
            elif outcome == "ineligible":
                summary["totals"]["ineligible_generations"] += 1
            elif outcome == "check_error":
                summary["totals"]["check_error_generations"] += 1

            if outcome == "passed":
                summary["totals"]["overall_pass_count"] += 1

            metrics = r.get("metrics", {})
            cost = metrics.get("cost")
            if cost is None:
                summary["totals"]["unknown_cost_generations"] += 1
            else:
                summary["totals"]["total_cost"] += cost
                summary["totals"]["known_cost_generations"] += 1

            attempt_latency = metrics.get("attempt_latency", metrics.get("api_latency"))
            if attempt_latency is not None:
                summary["totals"]["total_time"] += attempt_latency
            successful_latency = metrics.get("api_latency")
            if successful_latency is not None:
                summary["totals"]["successful_latency_total"] += successful_latency
                summary["totals"]["successful_latency_count"] += 1

            # By model
            model = r["model"]
            if model not in summary["by_model"]:
                summary["by_model"][model] = {
                    "provider": r["provider"],
                    "tested": 0,
                    "passed": 0,
                    "failed": 0,
                    "generation_errors": 0,
                    "rate_limited": 0,
                    "check_errors": 0,
                    "validation_failed": 0,
                    "ineligible": 0,
                    "eligible": 0,
                    "pass_rate": 0.0,
                    "total_cost": 0.0,
                    "known_cost_generations": 0,
                    "unknown_cost_generations": 0,
                    "total_latency": 0.0,
                    "successful_latency_count": 0,
                    "avg_latency": None,
                }
            m = summary["by_model"][model]
            m["tested"] += 1
            if outcome == "passed":
                m["passed"] += 1
            if r.get("error"):
                m["failed"] += 1
                if outcome == "rate_limited":
                    m["rate_limited"] += 1
                else:
                    m["generation_errors"] += 1
            elif outcome == "failed":
                m["validation_failed"] += 1
            elif outcome == "ineligible":
                m["ineligible"] += 1
            elif outcome == "check_error":
                m["check_errors"] += 1
            if outcome in {"passed", "failed"}:
                m["eligible"] += 1
            if cost is None:
                m["unknown_cost_generations"] += 1
            else:
                m["total_cost"] += cost
                m["known_cost_generations"] += 1
            if successful_latency is not None:
                m["total_latency"] += successful_latency
                m["successful_latency_count"] += 1

            # By root
            root = r["root"]
            if root not in summary["by_root"]:
                summary["by_root"][root] = {
                    "tested": 0,
                    "passed": 0,
                    "validation_failed": 0,
                    "generation_errors": 0,
                    "rate_limited": 0,
                    "check_errors": 0,
                    "ineligible": 0,
                    "eligible": 0,
                    "pass_rate": 0.0,
                }
            summary["by_root"][root]["tested"] += 1
            if outcome == "passed":
                summary["by_root"][root]["passed"] += 1
            elif outcome == "failed":
                summary["by_root"][root]["validation_failed"] += 1
            elif outcome == "generation_error":
                summary["by_root"][root]["generation_errors"] += 1
            elif outcome == "rate_limited":
                summary["by_root"][root]["rate_limited"] += 1
            elif outcome == "ineligible":
                summary["by_root"][root]["ineligible"] += 1
            elif outcome == "check_error":
                summary["by_root"][root]["check_errors"] += 1
            if outcome in {"passed", "failed"}:
                summary["by_root"][root]["eligible"] += 1

            # By scale
            scale = r["scale"]
            if scale not in summary["by_scale"]:
                summary["by_scale"][scale] = {
                    "tested": 0,
                    "passed": 0,
                    "validation_failed": 0,
                    "generation_errors": 0,
                    "rate_limited": 0,
                    "check_errors": 0,
                    "ineligible": 0,
                    "eligible": 0,
                    "pass_rate": 0.0,
                }
            summary["by_scale"][scale]["tested"] += 1
            if outcome == "passed":
                summary["by_scale"][scale]["passed"] += 1
            elif outcome == "failed":
                summary["by_scale"][scale]["validation_failed"] += 1
            elif outcome == "generation_error":
                summary["by_scale"][scale]["generation_errors"] += 1
            elif outcome == "rate_limited":
                summary["by_scale"][scale]["rate_limited"] += 1
            elif outcome == "ineligible":
                summary["by_scale"][scale]["ineligible"] += 1
            elif outcome == "check_error":
                summary["by_scale"][scale]["check_errors"] += 1
            if outcome in {"passed", "failed"}:
                summary["by_scale"][scale]["eligible"] += 1

        # Calculate rates
        eligible_total = summary["totals"]["eligible_generations"]
        if eligible_total > 0:
            summary["totals"]["overall_pass_rate"] = (
                summary["totals"]["overall_pass_count"] / eligible_total
            )

        for model, m in summary["by_model"].items():
            if m["eligible"] > 0:
                m["pass_rate"] = m["passed"] / m["eligible"]
            if m["successful_latency_count"]:
                m["avg_latency"] = m["total_latency"] / m["successful_latency_count"]

        if summary["totals"]["successful_latency_count"]:
            summary["totals"]["avg_successful_latency"] = (
                summary["totals"]["successful_latency_total"]
                / summary["totals"]["successful_latency_count"]
            )

        for root, r in summary["by_root"].items():
            if r["eligible"] > 0:
                r["pass_rate"] = r["passed"] / r["eligible"]

        for scale, s in summary["by_scale"].items():
            if s["eligible"] > 0:
                s["pass_rate"] = s["passed"] / s["eligible"]

        return summary

    def _sanitize_filename(self, text: str, max_len: int = 50) -> str:
        """
        Create safe filename from text.

        Args:
            text: Original text
            max_len: Maximum length of output

        Returns:
            str: Sanitized filename
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        suffix = f"-{digest}"
        if max_len <= len(suffix):
            raise ValueError("max_len must leave room for a readable path component")

        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in text)
        safe = safe.strip("._-") or "item"
        readable_length = max_len - len(suffix)
        safe = safe[:readable_length].rstrip("._-") or "item"
        return f"{safe}{suffix}"

    def _create_run_directory(self, run_name: str) -> tuple[Path, str, str]:
        """Create a collision-resistant directory and return its authoritative metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_id = f"{timestamp}_{self._sanitize_filename(run_name, max_len=32)}_{uuid4().hex[:16]}"
        run_path = self.output_dir / run_id
        run_path.mkdir(parents=True, exist_ok=False)
        return run_path, run_id, timestamp

    @staticmethod
    def _task_fingerprint(task: dict) -> str:
        """Return a stable digest for all task inputs that affect artifacts."""
        task_inputs = {
            key: task.get(key)
            for key in (
                "provider",
                "model",
                "original_prompt",
                "full_prompt",
                "root",
                "scale",
                "variation_name",
                "use_thinking",
                "effort",
                "test_params",
            )
        }
        canonical = json.dumps(task_inputs, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    """Run the guarded broad cloud-evaluation example."""

    if not confirm_direct_evaluation():
        raise SystemExit(1)

    eval = Evaluator(output_dir="runs", temperature=0.0)
    eval.evaluate(
        prompts=[
            "An arpeggiator in only quarter notes",
            "An arpeggiator in only eighth notes",
            "An arpeggiator in only sixteenth notes",
        ],
        roots=["C", "A", "F#", "Eb"],
        models=[
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gemini-3.5-flash",
            "gemini-3.1-pro",
            "gemini-3.1-flash-lite",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-opus-4-5",
        ],
        run_name="top cloud models",
        tests=["scale", "duration"],
        test_reasoning=True,
    )


if __name__ == "__main__":
    main()

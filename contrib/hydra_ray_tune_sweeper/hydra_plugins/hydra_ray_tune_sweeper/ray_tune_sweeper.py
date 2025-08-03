# hydra_ray_tune_sweeper/hydra_plugins/hydra_ray_tune_sweeper/ray_tune_sweeper.py
"""Ray Tune sweeper implementation.

This module provides the main RayTuneSweeper class that integrates
Ray Tune's optimization algorithms with Hydra's configuration system.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ray
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, JobStatus, run_job, setup_globals
from hydra.errors import HydraException
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf, open_dict
from ray import tune
from ray.tune.schedulers import TrialScheduler
from ray.tune.search import SearchAlgorithm

from ._core import create_search_space

log = logging.getLogger(__name__)


class RayTuneSweeper(Sweeper):
    """Ray Tune based hyperparameter sweeper for Hydra.

    This sweeper integrates Ray Tune's optimization algorithms and distributed
    execution capabilities with Hydra's configuration management system.

    IMPORTANT: This sweeper manages its own execution and is incompatible with
    other Hydra launchers (ray, submitit, etc.). It only works with the basic
    launcher since Ray Tune handles the distributed execution internally.
    """

    def __init__(
        self,
        num_samples: int = 10,
        max_concurrent_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        search_alg: Optional[SearchAlgorithm] = None,
        scheduler: Optional[TrialScheduler] = None,
        ray_config: Optional[Dict[str, Any]] = None,
        resume: Union[bool, str] = False,
        max_failures: int = 0,
        fail_fast: bool = False,
        resources_per_trial: Optional[Dict[str, Union[int, float]]] = None,
        metric: Optional[str] = None,
        mode: str = "min",
        local_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        params: Optional[Dict[str, str]] = None,
        stop: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.num_samples = num_samples
        self.max_concurrent_trials = max_concurrent_trials
        self.timeout = timeout
        self.search_alg = search_alg
        self.scheduler = scheduler
        self.ray_config = ray_config or {}
        self.resume = resume
        self.max_failures = max_failures
        self.fail_fast = fail_fast
        self.resources_per_trial = resources_per_trial or {"cpu": 1, "gpu": 0}
        self.metric = metric
        self.mode = mode
        self.local_dir = local_dir
        self.experiment_name = experiment_name
        self.params = params or {}
        self.stop = stop

        # Standard Hydra sweeper attributes
        self.config: Optional[DictConfig] = None
        self.hydra_context: Optional[HydraContext] = None
        self.launcher: Optional[Any] = None
        self.task_function: Optional[TaskFunction] = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        """Set up the Ray Tune sweeper using standard Hydra patterns."""
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

        # Validate that we're using a compatible launcher
        self._validate_plugin_compatibility(config)

        # Since Ray Tune manages execution, we don't need a launcher
        # but we store it for potential future compatibility
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, hydra_context=hydra_context, task_function=task_function
        )

        # Initialize Ray if not already running
        if not ray.is_initialized():
            log.info("Initializing Ray for optimization")
            ray.init(**self.ray_config)

    def _validate_plugin_compatibility(self, config: DictConfig) -> None:
        """Validate that the launcher is compatible with Ray Tune sweeper.

        Ray Tune manages its own distributed execution, so it's incompatible
        with other launchers like ray, submitit, joblib, etc.
        """
        launcher_config = config.get("hydra", {}).get("launcher", None)

        if launcher_config is None:
            return  # No launcher specified, use default

        if hasattr(launcher_config, "_target_"):
            launcher_target = launcher_config._target_

            # Only basic launcher is compatible
            compatible_launchers = [
                "hydra._internal.core_plugins.basic_launcher.BasicLauncher",
            ]

            if launcher_target not in compatible_launchers:
                raise HydraException(
                    f"Ray Tune sweeper is incompatible with launcher '{launcher_target}'.\n"
                    f"Ray Tune manages its own distributed execution and only works with the basic launcher.\n"
                    f"Please remove 'hydra/launcher' from your config or use 'hydra/launcher=basic'."
                )

    def sweep(self, arguments: List[str]) -> None:
        """Execute hyperparameter sweep using Ray Tune for optimization strategy."""
        assert self.config is not None
        assert self.hydra_context is not None
        assert self.task_function is not None

        # Call setup_globals() like other launchers do - this ensures resolvers are registered
        setup_globals()

        # Debug: Print minimal config info for troubleshooting
        log.debug(f"Sweep dir from config: {self.config.hydra.sweep.dir}")

        # Parse command line arguments and plugin config
        params_conf = self._parse_config()
        params_conf.extend(arguments)

        if not params_conf:
            log.warning("No parameters specified for optimization")
            return

        parser = OverridesParser.create(config_loader=self.hydra_context.config_loader)
        overrides = parser.parse_overrides(params_conf)

        # Create search space from overrides
        search_space = create_search_space(overrides)
        if not search_space:
            log.warning("No sweep parameters found - nothing to optimize")
            return

        # Handle edge case where config.hydra.sweep.dir might not be set properly
        # This can happen in testing scenarios with complex override configurations
        sweep_dir_value = self.config.hydra.sweep.dir
        if sweep_dir_value is None or str(sweep_dir_value) == "None":
            # Check if there's a sweep directory in the overrides
            for override in self.config.hydra.overrides.hydra:
                if override.startswith("hydra.sweep.dir=") and not override.endswith("=None"):
                    # Extract the directory path from the override
                    sweep_dir_value = override.split("=", 1)[1]
                    log.debug(f"Found sweep dir in overrides: {sweep_dir_value}")

            if sweep_dir_value is None:
                # Fallback to default directory structure
                sweep_dir_value = "./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"
                log.warning(f"hydra.sweep.dir is None, using default: {sweep_dir_value}")

        sweep_dir = Path(str(sweep_dir_value))
        sweep_dir.mkdir(parents=True, exist_ok=True)

        # Store the corrected sweep directory for use in _save_results
        self._resolved_sweep_dir = sweep_dir

        # Save sweep run config like other sweepers do
        OmegaConf.save(self.config, sweep_dir / "multirun.yaml")

        log.info(f"Ray Tune Launcher is launching {len(search_space)} parameter sweeps")
        log.info(f"Sweep output dir: {sweep_dir}")

        # Set up Ray Tune components
        search_alg = self.search_alg
        scheduler = self.scheduler

        log.info(f"Starting Ray Tune optimization with {self.num_samples} trials")
        log.info(f"Search space: {search_space}")

        # Track job index for proper hydra.job.num assignment
        self._job_idx = 0

        # Use Ray Tune's built-in function API with Hydra task execution
        def objective(config: Dict[str, Any]) -> Dict[str, Any]:
            """Objective function that runs Hydra task with given config."""
            return self._run_hydra_task_directly(config)

        # Use the sweep directory that we just created and verified
        # But handle the case where self.local_dir is preferred
        storage_path = self.local_dir or str(sweep_dir.absolute())

        # Configure and run Ray Tune experiment
        run_config = tune.RunConfig(
            name=self.experiment_name or "hydra_ray_tune_sweep",
            storage_path=storage_path,
            failure_config=tune.FailureConfig(
                max_failures=self.max_failures,
                fail_fast=self.fail_fast,
            ),
            stop=self.stop,
        )

        tune_config = tune.TuneConfig(
            num_samples=self.num_samples,
            max_concurrent_trials=self.max_concurrent_trials,
            time_budget_s=self.timeout,
            search_alg=search_alg,
            scheduler=scheduler,
            metric=self.metric,
            mode=self.mode,
        )

        tuner = tune.Tuner(
            objective,
            param_space=search_space,
            tune_config=tune_config,
            run_config=run_config,
        )

        try:
            results = tuner.fit()
            self._save_results(results)
        except Exception as e:
            log.error(f"Ray Tune experiment failed: {e}")
            raise

    def _run_hydra_task_directly(self, tune_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run Hydra task directly without using launcher.

        Since Ray Tune manages the distributed execution, we bypass the launcher
        and run the task directly in the Ray worker process.
        """
        # Setup globals for Hydra
        setup_globals()

        # Convert Ray Tune config to Hydra overrides
        overrides = [f"{k!s}={v}" for k, v in tune_config.items()]

        # Load sweep configuration with overrides
        sweep_config = self.hydra_context.config_loader.load_sweep_config(self.config, overrides)

        # Set job metadata
        with open_dict(sweep_config):
            sweep_config.hydra.job.id = f"ray_tune_{self._job_idx}"
            sweep_config.hydra.job.num = self._job_idx

            # Set output directory for this trial
            trial_dir = Path(str(self.config.hydra.sweep.dir)) / f"{self._job_idx}"
            sweep_config.hydra.runtime.output_dir = str(trial_dir)

        # Set singleton state for this trial
        Singleton.set_state(Singleton.get_state())

        # Set Hydra config
        HydraConfig.instance().set_config(sweep_config)

        # Increment job index for next trial
        self._job_idx += 1

        try:
            # Run the task function directly
            ret = run_job(
                hydra_context=self.hydra_context,
                task_function=self.task_function,
                config=sweep_config,
                job_dir_key="hydra.runtime.output_dir",
                job_subdir_key=None,
            )

            # Handle job failure
            if ret.status == JobStatus.FAILED:
                if isinstance(ret.return_value, Exception):
                    raise ret.return_value
                else:
                    raise RuntimeError(f"Task failed with return value: {ret.return_value}")

            # Extract metrics from return value
            return self._extract_metrics(ret.return_value)

        except Exception as e:
            log.error(f"Trial {self._job_idx - 1} failed with error: {e}")
            raise

    def _extract_metrics(self, return_value: Any) -> Dict[str, float]:
        """Extract metrics from task return value for Ray Tune."""
        if isinstance(return_value, dict):
            # Filter to numeric values only
            metrics = {}
            for k, v in return_value.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
            return metrics
        elif isinstance(return_value, (int, float)):
            return {"objective": float(return_value)}
        elif return_value is None:
            return {"objective": 0.0}
        else:
            # Try to convert to float, fallback to 0
            try:
                return {"objective": float(return_value)}
            except (ValueError, TypeError):
                log.warning(f"Cannot convert return value to metric: {return_value}")
                return {"objective": 0.0}

    def _parse_config(self) -> List[str]:
        """Parse sweeper parameter configuration."""
        params_conf = []
        for k, v in self.params.items():
            params_conf.append(f"{k}={v}")
        return params_conf

    def _save_results(self, results: "tune.ResultGrid") -> None:
        """Save optimization results following Hydra patterns."""
        try:
            # Get best result
            if self.metric:
                best_result = results.get_best_result(metric=self.metric, mode=self.mode)
                best_config = best_result.config
                best_metrics = best_result.metrics
            else:
                # Use first result if no metric specified
                results_df = results.get_dataframe()
                if len(results_df) > 0:
                    best_config = results_df.iloc[0].to_dict()
                    best_metrics = {}
                else:
                    log.warning("No results found")
                    return

            results_to_serialize = {
                "name": "ray_tune",
                "best_config": best_config,
                "best_metrics": best_metrics,
                "total_trials": len(results),
            }

            if self.metric and self.metric in best_metrics:
                results_to_serialize["best_value"] = best_metrics[self.metric]

            # Use the exact same pattern as Optuna sweeper
            # Use the resolved sweep directory instead of the potentially incorrect config value
            results_file = self._resolved_sweep_dir / "optimization_results.yaml"
            OmegaConf.save(
                OmegaConf.create(results_to_serialize),
                results_file,
            )

            log.info(f"Best config: {best_config}")
            log.info(f"Results saved to: {results_file}")

        except Exception as e:
            log.error(f"Failed to save results: {e}")

# hydra_ray_tune_sweeper/hydra_plugins/hydra_ray_tune_sweeper/ray_tune_sweeper.py

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ray
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from hydra.core.utils import JobReturn, JobStatus
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ray import tune
from ray.tune.schedulers import TrialScheduler
from ray.tune.search import SearchAlgorithm

from ._core import create_search_space

log = logging.getLogger(__name__)


class RayTuneSweeper(Sweeper):
    """Ray Tune sweeper for Hydra applications.

    This sweeper integrates Ray Tune's optimization algorithms and distributed
    execution capabilities with Hydra's configuration management system.

    Unlike other Hydra sweepers, this plugin combines both sweeping strategy
    and execution management, making it incompatible with other launchers.
    """

    def __init__(
        self,
        num_samples: int = 10,
        max_concurrent_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        search_alg: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
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
        self.search_alg_config = search_alg
        self.scheduler_config = scheduler
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

        # Use Hydra's standard launcher instantiation pattern
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, hydra_context=hydra_context, task_function=task_function
        )

        # Initialize Ray if not already running
        if not ray.is_initialized():
            log.info("Initializing Ray for optimization")
            ray.init(**self.ray_config)

    def sweep(self, arguments: List[str]) -> None:
        """Execute hyperparameter sweep using Ray Tune for optimization strategy."""
        assert self.config is not None
        assert self.hydra_context is not None
        assert self.task_function is not None
        assert self.launcher is not None

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

        # Debug the config state before using it
        log.info(f"Raw config.hydra.sweep.dir: {repr(self.config.hydra.sweep.dir)}")
        log.info(f"Type of config.hydra.sweep.dir: {type(self.config.hydra.sweep.dir)}")
        log.info(f"self.local_dir: {repr(self.local_dir)}")

        # Follow standard Hydra sweeper pattern - create sweep directory immediately
        sweep_dir_raw = self.config.hydra.sweep.dir
        log.info(f"Before str() conversion: {repr(sweep_dir_raw)}")

        sweep_dir_str = str(sweep_dir_raw)
        log.info(f"After str() conversion: {repr(sweep_dir_str)}")

        sweep_dir = Path(sweep_dir_str)
        log.info(f"Path object: {repr(sweep_dir)}")
        log.info(f"Absolute path: {repr(sweep_dir.absolute())}")

        sweep_dir.mkdir(parents=True, exist_ok=True)

        # Save sweep run config like other sweepers do
        OmegaConf.save(self.config, sweep_dir / "multirun.yaml")

        log.info(f"Ray Tune Launcher is launching {len(search_space)} parameter sweeps")
        log.info(f"Sweep output dir: {sweep_dir}")

        # Set up Ray Tune components
        search_alg = self._create_search_algorithm()
        scheduler = self._create_scheduler()

        log.info(f"Starting Ray Tune optimization with {self.num_samples} trials")
        log.info(f"Search space: {search_space}")

        # Use Ray Tune's built-in function API with Hydra task execution
        def objective(config: Dict[str, Any]) -> Dict[str, Any]:
            """Objective function that runs Hydra task with given config."""
            return self._run_hydra_task(config)

        # Use the sweep directory that we just created and verified
        # But handle the case where self.local_dir is preferred
        if self.local_dir:
            storage_path = self.local_dir
        else:
            storage_path = str(sweep_dir.absolute())
        log.info(f"Final storage_path being passed to Ray Tune: {repr(storage_path)}")

        # Additional debugging - let's also check what self.config looks like
        log.info("=== DEBUGGING CONFIG STATE ===")
        log.info(f"self.config type: {type(self.config)}")
        log.info(f"self.config.hydra type: {type(self.config.hydra)}")
        log.info(f"self.config.hydra.sweep type: {type(self.config.hydra.sweep)}")

        # Try to serialize parts of the config to see what's resolved
        try:
            sweep_config = OmegaConf.to_container(self.config.hydra.sweep, resolve=True)
            log.info(f"Resolved sweep config: {sweep_config}")
        except Exception as e:
            log.error(f"Failed to resolve sweep config: {e}")

        try:
            hydra_config = OmegaConf.to_container(self.config.hydra, resolve=True)
            log.info(
                f"Resolved hydra config keys: {list(hydra_config.keys()) if isinstance(hydra_config, dict) else 'Not a dict'}"
            )
        except Exception as e:
            log.error(f"Failed to resolve hydra config: {e}")
        log.info("=== END DEBUGGING ===")

        # Configure and run Ray Tune experiment
        log.info("=== CREATING RAY TUNE TUNER ===")
        log.info(f"storage_path: {repr(storage_path)}")
        log.info(f"experiment_name: {repr(self.experiment_name)}")
        log.info(f"stop: {repr(self.stop)}")

        run_config = tune.RunConfig(
            name=self.experiment_name or "hydra_ray_tune_sweep",
            storage_path=storage_path,
            failure_config=tune.FailureConfig(
                max_failures=self.max_failures,
                fail_fast=self.fail_fast,
            ),
            stop=self.stop,
        )
        log.info(f"Created RunConfig: {run_config}")

        tune_config = tune.TuneConfig(
            num_samples=self.num_samples,
            max_concurrent_trials=self.max_concurrent_trials,
            time_budget_s=self.timeout,
            search_alg=search_alg,
            scheduler=scheduler,
            metric=self.metric,
            mode=self.mode,
        )
        log.info(f"Created TuneConfig: {tune_config}")

        log.info("About to create Tuner...")
        tuner = tune.Tuner(
            objective,
            param_space=search_space,
            tune_config=tune_config,
            run_config=run_config,
        )
        log.info("Tuner created successfully!")
        log.info("=== END TUNER CREATION ===")

        try:
            results = tuner.fit()
            self._save_results(results)
        except Exception as e:
            log.error(f"Ray Tune experiment failed: {e}")
            raise

    def _run_hydra_task(self, tune_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run Hydra task with Ray Tune configuration.

        This follows the pattern used by other Hydra sweepers - generate
        overrides and use the launcher to execute jobs.
        """
        # Convert Ray Tune config to Hydra overrides
        # Use + prefix to add new parameters that might not exist in base config
        overrides = [f"+{k}={v}" for k, v in tune_config.items()]

        # Use launcher to run single job (like BasicSweeper does)
        job_returns = self.launcher.launch([overrides], initial_job_idx=0)

        if not job_returns:
            raise RuntimeError("No job results returned")

        job_return = job_returns[0]

        # Handle job failure
        if job_return.status == JobStatus.FAILED:
            # Access return_value to trigger exception propagation
            _ = job_return.return_value

        # Extract metrics from return value
        return self._extract_metrics(job_return.return_value)

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

    def _create_search_algorithm(self) -> Optional[SearchAlgorithm]:
        """Create search algorithm from configuration."""
        if self.search_alg_config is None:
            return None

        try:
            search_alg = instantiate(self.search_alg_config)
            return search_alg
        except Exception as e:
            log.error(f"Failed to create search algorithm: {e}")
            return None

    def _create_scheduler(self) -> Optional[TrialScheduler]:
        """Create trial scheduler from configuration."""
        if self.scheduler_config is None:
            return None

        try:
            scheduler = instantiate(self.scheduler_config)
            return scheduler
        except Exception as e:
            log.error(f"Failed to create scheduler: {e}")
            return None

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

            # Prepare results for serialization (following Optuna sweeper pattern)
            results_to_serialize = {
                "name": "ray_tune",
                "best_config": best_config,
                "best_metrics": best_metrics,
                "total_trials": len(results),
            }

            if self.metric and self.metric in best_metrics:
                results_to_serialize["best_value"] = best_metrics[self.metric]

            # Save results
            results_path = Path(self.config.hydra.sweep.dir) / "optimization_results.yaml"
            OmegaConf.save(OmegaConf.create(results_to_serialize), results_path)

            log.info(f"Best config: {best_config}")
            log.info(f"Results saved to: {results_path}")

        except Exception as e:
            log.error(f"Failed to save results: {e}")

    def validate_batch_is_legal(self, batch) -> None:
        """Validate batch compatibility using parent implementation."""
        super().validate_batch_is_legal(batch)

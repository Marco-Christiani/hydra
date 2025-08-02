# hydra_ray_tune_sweeper/hydra_plugins/hydra_ray_tune_sweeper/ray_tune_sweeper.py

import logging
from typing import Any, Dict, List, Optional

import ray
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.errors import HydraException
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ray import tune
from ray.tune.schedulers import TrialScheduler
from ray.tune.search import SearchAlgorithm

from ._core import HydraTrainable, create_search_space
from .config import RayTuneSweeperConf

log = logging.getLogger(__name__)


class RayTuneSweeper(Sweeper):
    """Ray Tune sweeper for Hydra applications.

    This sweeper integrates Ray Tune's optimization algorithms and distributed
    execution capabilities with Hydra's configuration management system.

    Unlike other Hydra sweepers, this plugin combines both sweeping strategy
    and execution management, making it incompatible with other launchers.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.config: Optional[DictConfig] = None
        self.hydra_context: Optional[HydraContext] = None
        self.task_function: Optional[TaskFunction] = None
        self.tune_config = RayTuneSweeperConf(**kwargs)

        # Ray Tune manages both optimization and execution
        self.launcher: Optional[Any] = None  # Not used - Ray Tune handles execution

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        """Set up the Ray Tune sweeper.

        Args:
            hydra_context: Hydra context for configuration loading
            task_function: User's task function to optimize
            config: Hydra configuration
        """
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

        # Validate plugin compatibility
        self._validate_plugin_compatibility(config)

        # Initialize Ray if not already running
        if not ray.is_initialized():
            log.info("Initializing Ray for Ray Tune sweeper")
            ray.init(**self.tune_config.ray_config)
        else:
            log.info("Ray already initialized - using existing Ray cluster")

    def _validate_plugin_compatibility(self, config: DictConfig) -> None:
        """Validate that Ray Tune sweeper is not mixed with incompatible plugins.

        Args:
            config: Hydra configuration to validate

        Raises:
            HydraException: If incompatible plugins are detected
        """
        # Check if a non-default launcher is configured
        launcher_config = config.hydra.launcher
        if launcher_config is not None:
            launcher_target = launcher_config.get("_target_")
            if launcher_target and "basic_launcher" not in launcher_target:
                raise HydraException(
                    f"Ray Tune sweeper is incompatible with launcher: {launcher_target}. "
                    f"Ray Tune manages both optimization strategy and distributed execution. "
                    f"Please remove launcher configuration or use a different sweeper."
                )

        # Check for other sweeper-launcher combinations that might conflict
        sweeper_config = config.hydra.get("sweeper")
        if sweeper_config is not None:
            sweeper_target = sweeper_config.get("_target_")
            if sweeper_target and "ray_tune_sweeper" in sweeper_target:
                log.info("Ray Tune sweeper detected - will manage both optimization and execution")

    def sweep(self, arguments: List[str]) -> None:
        """Execute hyperparameter sweep using Ray Tune.

        Args:
            arguments: Command-line override arguments
        """
        assert self.config is not None
        assert self.hydra_context is not None
        assert self.task_function is not None

        # Parse command line arguments and plugin config
        params_conf = self._parse_config()
        params_conf.extend(arguments)

        if not params_conf:
            log.warning("No parameters specified for optimization")
            return

        parser = OverridesParser.create(config_loader=self.hydra_context.config_loader)
        overrides = parser.parse_overrides(params_conf)

        # Create Ray Tune search space
        search_space = create_search_space(overrides)

        if not search_space:
            log.warning("No sweep parameters found - nothing to optimize")
            return

        # Set up search algorithm
        search_alg = self._create_search_algorithm(search_space)

        # Set up scheduler
        scheduler = self._create_scheduler()

        # Create Hydra trainable
        trainable_cls = HydraTrainable.with_parameters(
            hydra_context=self.hydra_context,
            task_function=self.task_function,
            base_config=self.config,
        )

        # Configure experiment
        experiment_config = self._create_experiment_config(trainable_cls, search_space)

        # Run Ray Tune experiment
        log.info(f"Starting Ray Tune experiment with {self.tune_config.num_samples} trials")
        log.info(f"Search space: {search_space}")
        log.info(f"Search algorithm: {type(search_alg).__name__ if search_alg else 'Random'}")
        if scheduler:
            log.info(f"Scheduler: {type(scheduler).__name__}")

        try:
            print("experiment config".center(100, "-"))
            __import__("pprint").pprint(experiment_config)
            analysis = tune.run(
                **experiment_config,
                search_alg=search_alg,
                scheduler=scheduler,
                verbose=1,  # Show progress
            )

            # Save results
            self._save_results(analysis)

        except Exception as e:
            log.error(f"Ray Tune experiment failed: {e}")
            raise
        finally:
            # Cleanup Ray resources if we initialized Ray
            if self.tune_config.ray_config:
                try:
                    ray.shutdown()
                except Exception as e:
                    log.warning(f"Error shutting down Ray: {e}")

    def _create_experiment_config(self, trainable_cls: type, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Create Ray Tune experiment configuration.

        Args:
            trainable_cls: Configured HydraTrainable class
            search_space: Parameter search space

        Returns:
            Ray Tune experiment configuration dictionary
        """
        print("self.config".center(100, "-"))
        __import__("pprint").pprint(self.config)
        print("self.tune_config".center(100, "-"))
        __import__("pprint").pprint(self.tune_config)
        print("self.hydra_context".center(100, "-"))
        __import__("pprint").pprint(self.hydra_context)
        print("-" * 100)
        config = {
            "run_or_experiment": trainable_cls,
            "config": search_space,
            "num_samples": self.tune_config.num_samples,
            "resources_per_trial": self.tune_config.resources_per_trial,
            "checkpoint_freq": self.tune_config.checkpoint_freq,
            "checkpoint_at_end": self.tune_config.checkpoint_at_end,
            "max_failures": self.tune_config.max_failures,
            # "storage_path": "file://" + (self.tune_config.local_dir or self.config.hydra.sweep.dir),
            "storage_path": self.tune_config.local_dir or self.config.hydra.sweep.dir,
            "name": self.tune_config.experiment_name or "hydra_ray_tune_sweep",
            "resume": self.tune_config.resume,
            "fail_fast": self.tune_config.fail_fast,
        }
        if not config["storage_path"]:
            del config["storage_path"]

        # Add optional parameters
        if self.tune_config.max_concurrent_trials:
            config["max_concurrent_trials"] = self.tune_config.max_concurrent_trials

        if self.tune_config.timeout:
            config["time_budget_s"] = self.tune_config.timeout

        if self.tune_config.stop:
            config["stop"] = self.tune_config.stop

        if self.tune_config.metric:
            config["metric"] = self.tune_config.metric
            config["mode"] = self.tune_config.mode

        return config

    def _create_search_algorithm(self, search_space: Dict[str, Any]) -> Optional[SearchAlgorithm]:
        """Create search algorithm from configuration.

        Args:
            search_space: Parameter search space

        Returns:
            Configured search algorithm or None for random search
        """
        if self.tune_config.search_alg is None:
            return None

        # search_alg_config = OmegaConf.to_container(self.tune_config.search_alg, resolve=True)

        try:
            print(f"{type(self.tune_config)=} {self.tune_config=}")
            __import__("pprint").pprint(self.tune_config)
            if isinstance(self.tune_config.search_alg, DictConfig):
                search_alg: SearchAlgorithm = instantiate(self.tune_config.search_alg)
                assert isinstance(search_alg, SearchAlgorithm)
            else:
                assert isinstance(self.tune_config.search_alg, SearchAlgorithm)
                search_alg: SearchAlgorithm = self.tune_config.search_alg

            # Configure search space for algorithms that require it
            if hasattr(search_alg, "set_search_properties"):
                search_alg.set_search_properties(
                    metric=self.tune_config.metric, mode=self.tune_config.mode, config=search_space
                )

            return search_alg

        except Exception as e:
            log.error(f"Failed to create search algorithm: {e}")
            log.warning("Falling back to random search")
            return None

    def _create_scheduler(self) -> Optional[TrialScheduler]:
        """Create trial scheduler from configuration.

        Returns:
            Configured trial scheduler or None for FIFO scheduling
        """
        if self.tune_config.scheduler is None:
            return None

        try:
            if isinstance(self.tune_config.scheduler, DictConfig):
                sched: TrialScheduler = instantiate(self.tune_config.scheduler)
                assert isinstance(sched, TrialScheduler)
            else:
                assert isinstance(self.tune_config.scheduler, TrialScheduler)
                sched: TrialScheduler = self.tune_config.scheduler
            return sched
        except Exception as e:
            log.error(f"Failed to create scheduler: {e}")
            log.warning("Falling back to FIFO scheduling")
            return None

    def _parse_config(self) -> List[str]:
        """Parse sweeper parameter configuration.

        Returns:
            List of parameter override strings
        """
        params_conf = []
        if self.tune_config.params:
            for k, v in self.tune_config.params.items():
                params_conf.append(f"{k}={v}")
        return params_conf

    def _save_results(self, analysis: "tune.Analysis") -> None:
        """Save optimization results.

        Args:
            analysis: Ray Tune analysis object with results
        """
        try:
            # Get best trial
            if self.tune_config.metric:
                best_trial = analysis.get_best_trial(metric=self.tune_config.metric, mode=self.tune_config.mode)
                best_config = analysis.get_best_config(metric=self.tune_config.metric, mode=self.tune_config.mode)
            else:
                # Use first completed trial if no metric specified
                completed_trials = [t for t in analysis.trials if t.status == "TERMINATED"]
                if completed_trials:
                    best_trial = completed_trials[0]
                    best_config = best_trial.config
                else:
                    log.warning("No completed trials found")
                    return

            best_result = best_trial.last_result if best_trial else {}

            # Prepare results for serialization
            results_to_serialize = {
                "name": "ray_tune",
                "best_config": best_config,
                "best_result": {
                    k: v for k, v in best_result.items() if isinstance(v, (int, float, str, bool, type(None)))
                },
                "experiment_path": analysis.experiment_path,
                "total_trials": len(analysis.trials),
                "completed_trials": len([t for t in analysis.trials if t.status == "TERMINATED"]),
                "failed_trials": len([t for t in analysis.trials if t.status == "ERROR"]),
            }

            # Save results to sweep directory
            results_path = f"{self.config.hydra.sweep.dir}/optimization_results.yaml"
            OmegaConf.save(OmegaConf.create(results_to_serialize), results_path)

            # Log summary
            log.info(f"Ray Tune experiment completed")
            log.info(f"Best config: {best_config}")
            log.info(f"Total trials: {len(analysis.trials)}")
            log.info(f"Results saved to: {results_path}")

            if self.tune_config.metric and self.tune_config.metric in best_result:
                log.info(f"Best {self.tune_config.metric}: {best_result[self.tune_config.metric]}")

        except Exception as e:
            log.error(f"Failed to save results: {e}")

    def validate_batch_is_legal(self, batch) -> None:
        """Validate job batch compatibility.

        Since Ray Tune manages its own execution, this method is not used
        but is required by the Sweeper interface.
        """
        # Ray Tune handles validation internally
        pass

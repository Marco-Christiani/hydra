# hydra_ray_tune_sweeper/hydra_plugins/hydra_ray_tune_sweeper/_tune_integration.py

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.types import ChoiceSweep, IntervalSweep, Override, RangeSweep, Transformer
from hydra.core.utils import JobStatus, run_job, setup_globals
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, open_dict
from ray import tune
from ray.tune import Trainable
from ray.tune.search.sample import Domain

log = logging.getLogger(__name__)


def create_tune_search_space_from_override(override: Override) -> Union[Domain, Any]:
    """Convert Hydra override to Ray Tune search space element.

    Args:
        override: Hydra override object

    Returns:
        Ray Tune search space element (Domain or fixed value)
    """
    if not override.is_sweep_override():
        # Fixed parameter - return the actual value
        return override.value()

    value = override.value()

    if override.is_choice_sweep():
        assert isinstance(value, ChoiceSweep)
        choices = [x for x in override.sweep_iterator(transformer=Transformer.encode)]
        return tune.choice(choices)

    elif override.is_range_sweep():
        assert isinstance(value, RangeSweep)

        if value.shuffle:
            # Convert shuffled range to choice
            choices = list(override.sweep_iterator())
            return tune.choice(choices)

        # For ranges, we need to handle the inclusive/exclusive upper bound difference
        # Ray Tune uses inclusive bounds, Hydra ranges are exclusive upper bound
        if isinstance(value.start, int) and isinstance(value.stop, int) and isinstance(value.step, int):
            # Integer range - make upper bound inclusive for Ray Tune
            upper_bound = value.stop - value.step if value.step > 0 else value.stop + abs(value.step)
            return tune.randint(lower=value.start, upper=upper_bound + 1)
        else:
            # Float range - convert to uniform distribution
            return tune.uniform(lower=float(value.start), upper=float(value.stop))

    elif override.is_interval_sweep():
        assert isinstance(value, IntervalSweep)

        if "log" in value.tags:
            return tune.loguniform(lower=value.start, upper=value.end)
        elif "int" in value.tags or (isinstance(value.start, int) and isinstance(value.end, int)):
            return tune.randint(lower=int(value.start), upper=int(value.end) + 1)
        else:
            return tune.uniform(lower=float(value.start), upper=float(value.end))

    else:
        raise ValueError(f"Unsupported sweep type: {override.value_type}")


def create_search_space(overrides: list[Override]) -> Dict[str, Union[Domain, Any]]:
    """Create Ray Tune search space from Hydra overrides.

    Args:
        overrides: List of parsed Hydra overrides

    Returns:
        Dictionary mapping parameter names to Ray Tune search space elements
    """
    search_space = {}

    for override in overrides:
        param_name = override.get_key_element()
        search_space[param_name] = create_tune_search_space_from_override(override)

    log.debug(f"Created search space: {search_space}")
    return search_space


class HydraTrainable(Trainable):
    """Ray Tune Trainable that wraps Hydra task functions.

    This class bridges Ray Tune's execution model with Hydra's configuration
    and task execution system.
    """

    def __init__(self, config: Dict[str, Any] = None, logger_creator=None):
        # Store Hydra-specific parameters before calling parent init
        self._hydra_context: Optional[HydraContext] = None
        self._task_function: Optional[TaskFunction] = None
        self._base_config: Optional[DictConfig] = None

        super().__init__(config, logger_creator)

    @classmethod
    def with_parameters(
        cls,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        base_config: DictConfig,
    ) -> Type["HydraTrainable"]:
        """Create a Trainable class configured with Hydra parameters.

        Args:
            hydra_context: Hydra context for configuration loading
            task_function: User's task function to execute
            base_config: Base Hydra configuration

        Returns:
            Configured HydraTrainable class
        """

        class ConfiguredHydraTrainable(cls):
            def __init__(self, config: Dict[str, Any] = None, logger_creator=None):
                super().__init__(config, logger_creator)

                # Store Hydra parameters
                self._hydra_context = hydra_context
                self._task_function = task_function
                self._base_config = base_config

        return ConfiguredHydraTrainable

    def setup(self, config: Dict[str, Any]) -> None:
        """Initialize the trainable with Ray Tune config.

        Args:
            config: Ray Tune trial configuration
        """
        if self._hydra_context is None:
            raise RuntimeError("HydraTrainable not properly configured. Use with_parameters() class method.")

        # Set up Hydra environment
        setup_globals()

        # Create overrides from Ray Tune config
        overrides = [f"{k}={v}" for k, v in config.items()]

        # Load sweep configuration
        self.sweep_config = self._hydra_context.config_loader.load_sweep_config(self._base_config, overrides)

        # Set job metadata
        with open_dict(self.sweep_config):
            self.sweep_config.hydra.job.id = f"ray_tune_{self.trial_id}"
            self.sweep_config.hydra.job.num = self.trial_id

            # Set trial-specific output directory
            trial_dir = Path(self.logdir)
            self.sweep_config.hydra.runtime.output_dir = str(trial_dir)

        # Set Hydra config
        HydraConfig.instance().set_config(self.sweep_config)

        log.debug(f"Trial {self.trial_id} setup with config: {config}")

    def step(self) -> Dict[str, Any]:
        """Execute one training step.

        For most Hydra tasks, this will be a single execution.
        For iterative tasks, this could be called multiple times.

        Returns:
            Dictionary of metrics for Ray Tune
        """
        # For single-shot optimization, run the full task once
        if not hasattr(self, "_executed"):
            result = self._run_task()
            self._executed = True

            # Mark as done for single-shot tasks
            result["done"] = True
            return result
        else:
            # Task already executed
            return {"done": True}

    def _run_task(self) -> Dict[str, Any]:
        """Execute the Hydra task function.

        Returns:
            Dictionary of metrics extracted from task return value
        """
        try:
            # Execute task using Hydra's run_job
            job_return = run_job(
                hydra_context=self._hydra_context,
                task_function=self._task_function,
                config=self.sweep_config,
                job_dir_key="hydra.runtime.output_dir",
                job_subdir_key=None,  # Already set in setup
                configure_logging=True,
            )

            # Extract metrics from return value
            return_value = job_return.return_value

            if job_return.status == JobStatus.FAILED:
                # Re-raise the exception to mark trial as failed
                if isinstance(return_value, Exception):
                    raise return_value
                else:
                    raise RuntimeError(f"Task failed with return value: {return_value}")

            # Convert return value to metrics
            metrics = self._extract_metrics(return_value)

            # Add metadata
            metrics.update({
                "trial_id": self.trial_id,
                "training_iteration": 1,
                "timesteps_total": 1,
            })

            log.debug(f"Trial {self.trial_id} completed with metrics: {metrics}")
            return metrics

        except Exception as e:
            log.error(f"Trial {self.trial_id} failed with error: {e}")
            # Re-raise to let Ray Tune handle the failure
            raise

    def _extract_metrics(self, return_value: Any) -> Dict[str, Any]:
        """Extract metrics from task return value.

        Args:
            return_value: Value returned by the task function

        Returns:
            Dictionary of metrics for Ray Tune
        """
        if isinstance(return_value, dict):
            # If task returns a dict, use it as metrics
            metrics = {}
            for k, v in return_value.items():
                if isinstance(v, (int, float, bool)):
                    metrics[k] = v
                else:
                    # Convert non-numeric values to strings
                    metrics[f"{k}_str"] = str(v)
            return metrics

        elif isinstance(return_value, (int, float)):
            # If task returns a scalar, use it as the primary objective
            return {"objective": return_value}

        elif return_value is None:
            # Task completed but returned None
            return {"objective": 0}

        else:
            # For other types, try to convert to float or use as string
            try:
                objective = float(return_value)
                return {"objective": objective}
            except (ValueError, TypeError):
                return {"objective": 0, "return_value_str": str(return_value)}

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        """Save checkpoint for resumable tasks.

        Args:
            checkpoint_dir: Directory to save checkpoint

        Returns:
            Path to saved checkpoint
        """
        # For most Hydra tasks, checkpointing would be task-specific
        # This is a basic implementation for compatibility
        checkpoint_path = os.path.join(checkpoint_dir, "hydra_checkpoint.txt")

        with open(checkpoint_path, "w") as f:
            f.write(f"trial_id: {self.trial_id}\n")
            f.write(f"executed: {hasattr(self, '_executed')}\n")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint for resumable tasks.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        # Basic implementation for compatibility
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                content = f.read()
                if "executed: True" in content:
                    self._executed = True

    def cleanup(self) -> None:
        """Clean up resources after trial completion."""
        # Cleanup any resources if needed
        pass

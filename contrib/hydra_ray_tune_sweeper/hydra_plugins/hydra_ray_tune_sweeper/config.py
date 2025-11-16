# hydra_ray_tune_sweeper/hydra_plugins/hydra_ray_tune_sweeper/config.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from ray import data


@dataclass
class SearchAlgorithmConfig:
    """Base configuration for Ray Tune search algorithms."""

    _target_: str = MISSING


@dataclass
class RandomSearchConfig(SearchAlgorithmConfig):
    """Random search algorithm configuration."""

    _target_: str = "ray.tune.search.basic_variant.BasicVariantGenerator"
    random_state: Optional[int] = None


@dataclass
class HyperOptConfig(SearchAlgorithmConfig):
    """HyperOpt search algorithm configuration."""

    _target_: str = "ray.tune.search.hyperopt.HyperOptSearch"
    metric: Optional[str] = None
    mode: str = "min"
    points_to_evaluate: Optional[List[Dict[str, Any]]] = None
    n_initial_points: int = 20
    random_state_seed: Optional[int] = None


@dataclass
class OptunaSearchConfig(SearchAlgorithmConfig):
    """Optuna search algorithm configuration."""

    _target_: str = "ray.tune.search.optuna.OptunaSearch"
    metric: Optional[str] = None
    mode: str = "min"
    sampler_class: str = "optuna.samplers.TPESampler"
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AxSearchConfig(SearchAlgorithmConfig):
    """Ax search algorithm configuration."""

    _target_: str = "ray.tune.search.ax.AxSearch"
    metric: Optional[str] = None
    mode: str = "min"
    parameter_constraints: Optional[List[str]] = None
    outcome_constraints: Optional[List[str]] = None


@dataclass
class TrialSchedulerConfig:
    """Base configuration for Ray Tune trial schedulers."""

    _target_: str = MISSING


@dataclass
class FIFOSchedulerConfig(TrialSchedulerConfig):
    """FIFO scheduler configuration (no early stopping)."""

    _target_: str = "ray.tune.schedulers.FIFOScheduler"


@dataclass
class ASHASchedulerConfig(TrialSchedulerConfig):
    """Asynchronous Successive Halving Algorithm scheduler configuration."""

    _target_: str = "ray.tune.schedulers.ASHAScheduler"
    metric: Optional[str] = None
    mode: str = "min"
    max_t: int = 100
    grace_period: int = 1
    reduction_factor: int = 2
    brackets: int = 1


@dataclass
class MedianStoppingConfig(TrialSchedulerConfig):
    """Median stopping rule scheduler configuration."""

    _target_: str = "ray.tune.schedulers.MedianStoppingRule"
    metric: Optional[str] = None
    mode: str = "min"
    time_attr: str = "training_iteration"
    grace_period: int = 1
    min_samples_required: int = 3
    min_time_slice: int = 0


@dataclass
class PopulationBasedTrainingConfig(TrialSchedulerConfig):
    """Population Based Training scheduler configuration.

    See https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html#ray.tune.schedulers.PopulationBasedTraining
    """

    _target_: str = "ray.tune.schedulers.PopulationBasedTraining"
    metric: Optional[str] = None
    mode: str = "min"
    time_attr: str = "training_iteration"
    perturbation_interval: int = 60
    burn_in_period: int = 0
    hyperparam_mutations: Dict[str, Any] = field(default_factory=dict)
    quantile_fraction: float = 0.25
    resample_probability: float = 0.25
    custom_explore_fn: Optional[str] = None


@dataclass
class RunConfig:
    """Ray Tune RunConfig.

    See https://docs.ray.io/en/latest/tune/api/doc/ray.tune.RunConfig.html#ray.tune.RunConfig
    """

    _target_: str = "ray.tune.RunConfig"


@dataclass
class RayTuneSweeperConf:
    """Ray Tune sweeper configuration.

    This sweeper combines both optimization strategy and execution management,
    making it incompatible with other Hydra sweepers and launchers.
    """

    _target_: str = "hydra_plugins.hydra_ray_tune_sweeper.ray_tune_sweeper.RayTuneSweeper"

    # Core Ray Tune parameters
    num_samples: int = 10
    max_concurrent_trials: Optional[int] = None
    timeout: Optional[float] = None

    # Search algorithm configuration
    search_alg: SearchAlgorithmConfig = field(default_factory=RandomSearchConfig)

    # Trial scheduler configuration
    scheduler: Optional[TrialSchedulerConfig] = None

    # Ray configuration
    ray_config: Dict[str, Any] = field(default_factory=dict)

    run_config: Optional[RunConfig] = None

    # Checkpointing and resumption
    resume: Union[bool, str] = False
    checkpoint_freq: int = 0
    checkpoint_at_end: bool = False

    # Failure handling
    # max_failures: int = 0
    # fail_fast: bool = False

    # Resource specification per trial
    resources_per_trial: Dict[str, Union[int, float]] = field(default_factory=lambda: {"cpu": 1, "gpu": 0})

    # Metric configuration
    metric: str = "objective"
    mode: str = "min"

    # Output configuration
    # local_dir: Optional[str] = None
    # experiment_name: Optional[str] = None

    # Hydra-specific configuration
    params: Optional[Dict[str, str]] = None

    # Trial stop condition
    # stop: Optional[Dict[str, Any]] = None


# Register configurations
ConfigStore.instance().store(
    group="hydra/sweeper", name="ray_tune", node=RayTuneSweeperConf, provider="ray_tune_sweeper"
)

# Search algorithm configurations
ConfigStore.instance().store(
    group="hydra/sweeper/search_alg", name="random", node=RandomSearchConfig, provider="ray_tune_sweeper"
)

ConfigStore.instance().store(
    group="hydra/sweeper/search_alg", name="hyperopt", node=HyperOptConfig, provider="ray_tune_sweeper"
)

ConfigStore.instance().store(
    group="hydra/sweeper/search_alg", name="optuna", node=OptunaSearchConfig, provider="ray_tune_sweeper"
)

ConfigStore.instance().store(
    group="hydra/sweeper/search_alg", name="ax", node=AxSearchConfig, provider="ray_tune_sweeper"
)

# Scheduler configurations
ConfigStore.instance().store(
    group="hydra/sweeper/scheduler", name="fifo", node=FIFOSchedulerConfig, provider="ray_tune_sweeper"
)

ConfigStore.instance().store(
    group="hydra/sweeper/scheduler", name="asha", node=ASHASchedulerConfig, provider="ray_tune_sweeper"
)

ConfigStore.instance().store(
    group="hydra/sweeper/scheduler", name="median_stopping", node=MedianStoppingConfig, provider="ray_tune_sweeper"
)

ConfigStore.instance().store(
    group="hydra/sweeper/scheduler", name="pbt", node=PopulationBasedTrainingConfig, provider="ray_tune_sweeper"
)

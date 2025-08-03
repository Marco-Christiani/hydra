# hydra_ray_tune_sweeper/tests/test_ray_tune_sweeper.py

import tempfile
from pathlib import Path
from typing import Dict

import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from hydra.errors import HydraException
from hydra.plugins.sweeper import Sweeper
from hydra.test_utils.test_utils import (
    TSweepRunner,
    chdir_plugin_root,
)
from hydra_plugins.hydra_ray_tune_sweeper._core import (
    create_search_space,
    create_tune_search_space_from_override,
)
from hydra_plugins.hydra_ray_tune_sweeper.ray_tune_sweeper import RayTuneSweeper
from omegaconf import DictConfig, OmegaConf
from ray.tune.search.sample import Categorical, Float, Integer

chdir_plugin_root()


def test_discovery() -> None:
    """Test that the plugin can be discovered via the plugins subsystem."""
    assert RayTuneSweeper.__name__ in [x.__name__ for x in Plugins.instance().discover(Sweeper)]


def test_override_to_tune_conversion() -> None:
    """Test conversion of Hydra overrides to Ray Tune search spaces."""
    parser = OverridesParser.create()

    # Test choice sweep
    choice_override = parser.parse_overrides(["param=choice(1,2,3)"])[0]
    choice_space = create_tune_search_space_from_override(choice_override)
    # corresponds to tune.choice()
    assert isinstance(choice_space, Categorical)

    # Test range sweep
    range_override = parser.parse_overrides(["param=range(1,5)"])[0]
    range_space = create_tune_search_space_from_override(range_override)
    # corresponds to tune.randint()
    assert isinstance(range_space, Integer)

    # Test float range sweep
    range_override = parser.parse_overrides(["param=range(1.0,5.0)"])[0]
    range_space = create_tune_search_space_from_override(range_override)
    # corresponds to tune.uniform()
    assert isinstance(range_space, Float)

    # Test interval sweep
    interval_override = parser.parse_overrides(["param=interval(0.1,1.0)"])[0]
    interval_space = create_tune_search_space_from_override(interval_override)
    # corresponds to tune.uniform()
    assert isinstance(interval_space, Float)

    # Test log interval
    log_override = parser.parse_overrides(["param=tag(log,interval(1e-4,1e-1))"])[0]
    log_space = create_tune_search_space_from_override(log_override)
    # corresponds to tune.loguniform()
    assert isinstance(log_space, Float)

    # Test fixed parameter
    fixed_override = parser.parse_overrides(["param=42"])[0]
    fixed_space = create_tune_search_space_from_override(fixed_override)
    assert fixed_space == 42


def test_search_space_creation() -> None:
    """Test creation of complete search space from multiple overrides."""
    parser = OverridesParser.create()
    overrides = parser.parse_overrides([
        "lr=interval(1e-4,1e-1)",
        "batch_size=choice(16,32,64)",
        "dropout=0.5",  # Fixed parameter
        "hidden_size=range(64,512)",
    ])

    search_space = create_search_space(overrides)

    assert len(search_space) == 4
    assert "lr" in search_space
    assert "batch_size" in search_space
    assert "dropout" in search_space
    assert "hidden_size" in search_space

    # Fixed parameter should be the actual value
    assert search_space["dropout"] == 0.5


def test_launcher_compatibility_validation() -> None:
    """Test that Ray Tune sweeper validates launcher compatibility."""
    sweeper = RayTuneSweeper()

    # Should raise error with non-basic launcher
    incompatible_config = OmegaConf.create({
        "hydra": {
            "launcher": {"_target_": "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"},
            # "sweeper": {"_target_": "hydra._internal.core_plugins.basic_sweeper.BasicSweeper"},
        }
    })

    with pytest.raises(HydraException, match="Ray Tune sweeper is incompatible with launcher"):
        sweeper._validate_plugin_compatibility(incompatible_config)

    # Should not raise error with basic launcher
    compatible_config = OmegaConf.create({
        "hydra": {"launcher": {"_target_": "hydra._internal.core_plugins.basic_launcher.BasicLauncher"}}
    })

    # Should not raise
    sweeper._validate_plugin_compatibility(compatible_config)

    # Should not raise error with no launcher specified
    no_launcher_config = OmegaConf.create({"hydra": {"launcher": None}})

    sweeper._validate_plugin_compatibility(no_launcher_config)


def simple_optimization_task(cfg: DictConfig) -> float:
    """Simple function for testing optimization."""
    x = cfg.x
    y = cfg.y
    return x**2 + y**2


# @pytest.mark.parametrize("search_alg", ["random", "hyperopt"])
@pytest.mark.parametrize("search_alg", ["random"])
def test_launch(search_alg: str, hydra_sweep_runner: TSweepRunner) -> None:
    """Test basic optimization with different search algorithms."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        overrides = [
            f"hydra.sweep.dir={tmp_dir}",
            "hydra/sweeper=ray_tune",
            # "hydra/sweeper=RayTuneSweeper",
            # f"hydra/sweeper/search_alg={search_alg}", # FIXME: whats the issue?
            "hydra.sweeper.num_samples=5",
            # "hydra.sweeper.metric=objective",
            "hydra.sweeper.mode=min",
            "hydra.sweeper.metric=null",
            "x=int(interval(0,4))",
            "y=int(interval(0,2))",
        ]
        print("Running with overrides:".center(100, "+"))
        __import__("pprint").pprint(overrides)
        print("+" * 100)
        sweep = hydra_sweep_runner(
            calling_file="examples/main.py",
            # calling_file=None,
            calling_module=None,
            # calling_module="hydra.test_utils.a_module",
            # calling_module="hydra_ray_tune_sweeper.tests.a_module",
            # calling_module="hydra_plugins.hydra_ray_tune_sweeper.tests.a_module",
            task_function=simple_optimization_task,
            # task_function=None,
            # config_path="configs",
            # config_name="compose.yaml",
            config_path="conf",
            config_name="config.yaml",
            overrides=overrides,
        )

        with sweep:
            assert sweep.returns is None  # Ray Tune manages execution

        # Check that results were saved
        # Since we arent optimizing anything we shouldnt see results
        for x in Path(tmp_dir).glob("**/*.yaml"):
            print(str(x))
        results_file = Path(tmp_dir) / "optimization_results.yaml"
        assert results_file.exists()

        print(results_file.read_text())

        results = OmegaConf.load(results_file)
        assert results.name == "ray_tune"
        print("Best Config".center(80, "-"))
        print(type(results))
        print(OmegaConf.to_yaml(results.best_config))
        print(type(results.best_config))
        assert "best_config" in results
        assert "x" in results.best_config
        assert "y" in results.best_config


def test_scheduler_integration() -> None:
    """Test integration with Ray Tune schedulers."""
    from ray.tune.schedulers import ASHAScheduler

    # Test with scheduler object directly
    asha_scheduler = ASHAScheduler(max_t=10, grace_period=1)
    sweeper = RayTuneSweeper(scheduler=asha_scheduler)

    assert sweeper.scheduler is not None
    assert sweeper.scheduler.__class__.__name__ == "AsyncHyperBandScheduler"


def test_config_parsing() -> None:
    """Test parsing of sweeper configuration."""
    sweeper = RayTuneSweeper(params={"lr": "interval(1e-4,1e-1)", "batch_size": "choice(16,32,64)"})

    params_conf = sweeper._parse_config()
    assert len(params_conf) == 2
    assert "lr=interval(1e-4,1e-1)" in params_conf
    assert "batch_size=choice(16,32,64)" in params_conf


def multi_metric_task(cfg: DictConfig) -> Dict[str, float]:
    """Task that returns multiple metrics."""
    x = cfg.x
    y = cfg.y

    return {
        "objective": (x - 2) ** 2 + (y - 1) ** 2,
        "constraint": x + y,
        "accuracy": 1.0 / (1.0 + (x - 2) ** 2 + (y - 1) ** 2),
    }


# def test_multi_metric_optimization(hydra_sweep_runner: TSweepRunner) -> None:
#     """Test optimization with multiple metrics."""
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         sweep = hydra_sweep_runner(
#             calling_file=None,
#             calling_module=None,
#             task_function=multi_metric_task,
#             config_path=None,
#             config_name=None,
#             overrides=[
#                 "hydra/sweeper=ray_tune",
#                 "hydra.sweeper.num_samples=3",
#                 "hydra.sweeper.metric=accuracy",
#                 "hydra.sweeper.mode=max",
#                 f"hydra.sweep.dir={tmp_dir}",
#                 "x=interval(0,4)",
#                 "y=interval(0,2)",
#             ],
#         )
#
#         with sweep:
#             pass
#
#         results_file = Path(tmp_dir) / "optimization_results.yaml"
#         assert results_file.exists()
#
#         results = OmegaConf.load(results_file)
#         assert "best_result" in results
#         assert "accuracy" in results.best_result

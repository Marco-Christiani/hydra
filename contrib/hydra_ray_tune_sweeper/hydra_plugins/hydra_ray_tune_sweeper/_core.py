# hydra_ray_tune_sweeper/hydra_plugins/hydra_ray_tune_sweeper/_tune_integration.py

import logging
from typing import Any, Dict, Union

from hydra.core.override_parser.types import ChoiceSweep, IntervalSweep, Override, RangeSweep, Transformer
from ray import tune
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

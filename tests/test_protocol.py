from dataclasses import asdict, replace

import pytest

from hvac_ppo.config import ExperimentConfig, ablation_configs, load_config
from hvac_ppo.io import ROOT
from hvac_ppo.scenarios import generate_cases


def test_ablation_changes_one_controlled_factor_at_a_time():
    base = load_config(ROOT / "configs/ppo/base.yaml")
    configs = ablation_configs(base, ROOT / "configs/ppo/ablation.yaml")
    assert len(configs) == 7
    for config in configs:
        differences = {key for key, value in asdict(base).items()
                       if key != "name" and value != asdict(config)[key]}
        assert len(differences) <= 1
        assert differences <= {"clip_range", "gae_lambda", "n_epochs"}


def test_fixed_splits_have_disjoint_physical_conditions():
    validation = generate_cases("validation", 1001, 5)["cases"]
    test = generate_cases("test", 2001, 10)["cases"]
    physical = lambda case: tuple(case[key] for key in
                                  ("scenario", "weather_delta_c", "indoor_init_c", "envelope_init_c"))
    assert not {physical(case) for case in validation} & {physical(case) for case in test}
    assert len(validation) == 15 and len(test) == 30


def test_partial_rollout_budgets_are_rejected():
    with pytest.raises(ValueError, match="complete rollouts"):
        replace(ExperimentConfig(), total_timesteps=100_000)

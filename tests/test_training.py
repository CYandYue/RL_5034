from dataclasses import replace

import numpy as np
import pytest
import torch

from hvac_ppo.config import ExperimentConfig
from hvac_ppo.evaluation import evaluate_cases, frozen_normalization
from hvac_ppo.io import read_json, write_json
from hvac_ppo.scenarios import load_cases
from hvac_ppo.training import complete_run, load_controller, train_run


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("ppo-roundtrip")
    cases = root / "cases.json"
    write_json(cases, {"schema_version": 1, "cases": [
        {"case_id": "test_hot_00", "scenario": "hot", "weather_delta_c": 0,
         "indoor_init_c": 25, "envelope_init_c": 30, "reset_seed": 42}]})
    config = ExperimentConfig(total_timesteps=64, eval_every=32, n_steps=32,
                              batch_size=16, n_epochs=2, environment={"episode_days": 1 / 24})
    output = root / "run"
    train_run(config, 7, output, cases, cases)
    return root, config, output, cases


def test_saved_policy_and_normalizer_reproduce_in_memory_predictions(trained):
    _, _, output, _ = trained
    controller, normalizer = load_controller(output)
    try:
        for probe in read_json(output / "metadata.json")["reload_probes"]:
            np.testing.assert_array_equal(controller(np.array(probe["observation"], dtype=np.float32)), probe["action"])
    finally:
        normalizer.close()


def test_evaluation_keeps_normalization_statistics_frozen(trained):
    _, config, output, cases = trained
    controller, normalizer = load_controller(output)
    before = (normalizer.obs_rms.mean.copy(), normalizer.obs_rms.var.copy(), normalizer.obs_rms.count)
    try:
        with frozen_normalization(normalizer):
            rows, _ = evaluate_cases(controller, load_cases(cases), config.environment)
        np.testing.assert_array_equal(normalizer.obs_rms.mean, before[0])
        np.testing.assert_array_equal(normalizer.obs_rms.var, before[1])
        assert normalizer.obs_rms.count == before[2]
        assert normalizer.training is False and normalizer.norm_reward is False
        assert rows == read_csv_as_metrics(output / "test_episodes.csv")
    finally:
        normalizer.close()


def read_csv_as_metrics(path):
    import csv
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["steps"] = int(row["steps"])
        for key in ("return_raw", "cvr_pct", "degree_hours", "energy_kwh"):
            row[key] = float(row[key])
    return rows


def test_completed_runs_are_verified_and_incompatible_runs_not_overwritten(trained):
    _, config, output, cases = trained
    assert complete_run(output)
    metadata_before = (output / "metadata.json").read_bytes()
    train_run(config, 7, output, cases, cases)  # Verified skip, not a second training.
    assert (output / "metadata.json").read_bytes() == metadata_before
    with pytest.raises(FileExistsError):
        train_run(replace(config, total_timesteps=96), 7, output, cases, cases)
    curve = output / "validation_curve.csv"
    original = curve.read_bytes()
    try:
        curve.write_bytes(original + b"\n")
        assert not complete_run(output)
    finally:
        curve.write_bytes(original)


def test_same_seed_repeats_cpu_training(trained):
    root, config, output, cases = trained
    repeated = root / "repeat"
    train_run(config, 7, repeated, cases, cases)
    a, norm_a = load_controller(output)
    b, norm_b = load_controller(repeated)
    try:
        for key, tensor in a.model.policy.state_dict().items():
            assert torch.equal(tensor, b.model.policy.state_dict()[key]), key
        assert read_json(output / "metadata.json")["test_mean"] == read_json(repeated / "metadata.json")["test_mean"]
    finally:
        norm_a.close()
        norm_b.close()

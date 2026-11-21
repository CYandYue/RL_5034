"""SB3 PPO with post-update validation, explicit artifacts and raw metrics."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import shutil
import time
import traceback

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hvac_env import HVAC2R2CEnv
from hvac_env.config import load_config as load_environment

from .config import ExperimentConfig, from_dict
from .evaluation import PPOController, evaluate_cases, frozen_normalization, mean_metrics
from .io import file_hash, git_info, read_json, stable_hash, training_source_hash, write_csv, write_json
from .scenarios import load_cases


class ObservedPPO(PPO):
    """Keep PPO's algorithm unchanged; observe only after a complete update."""

    def __init__(self, *args, **kwargs):
        self.update_observer = None
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["update_observer"]

    def train(self):
        super().train()
        if self.update_observer is not None:
            self.update_observer(self)
        self.logger.record("time/total_timesteps", self.num_timesteps)
        self.logger.dump(self.num_timesteps)


class ActionDiagnostics(BaseCallback):
    def _on_step(self):
        return True

    def _on_rollout_end(self):
        actions = self.model.rollout_buffer.actions
        self.logger.record("rollout/action_clip_fraction", float(np.mean(np.abs(actions) > 1)))


def make_normalizer(config: ExperimentConfig, seed: int) -> VecNormalize:
    env = DummyVecEnv([lambda: Monitor(HVAC2R2CEnv(config.environment)) for _ in range(config.n_envs)])
    env.seed(seed)
    return VecNormalize(env, norm_obs=config.normalize_observations, norm_reward=False,
                        clip_obs=config.clip_obs, gamma=config.gamma)


def configure_torch(threads: int):
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # Already configured when multiple smoke runs share one process.
    torch.use_deterministic_algorithms(True)


def run_identity(config: ExperimentConfig, seed: int, validation: Path, test: Path | None) -> str:
    return stable_hash({"config": config.snapshot(), "seed": seed,
                        "environment": asdict(load_environment(config.environment)),
                        "validation_sha256": file_hash(validation),
                        "test_sha256": file_hash(test) if test else None,
                        "training_source_hash": training_source_hash()})


def complete_run(path: Path, fingerprint: str | None = None) -> bool:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return False
    metadata = read_json(metadata_path)
    if metadata.get("status") != "complete":
        return False
    if fingerprint is not None and metadata.get("fingerprint") != fingerprint:
        return False
    for name, digest in metadata.get("artifacts", {}).items():
        artifact = path / name
        if not artifact.exists() or file_hash(artifact) != digest:
            return False
    return bool(metadata.get("artifacts"))


def train_run(config: ExperimentConfig, seed: int, output: Path, validation: Path,
              test: Path | None = None, restart: bool = False) -> dict:
    fingerprint = run_identity(config, seed, validation, test)
    if complete_run(output, fingerprint):
        return read_json(output / "metadata.json")
    if output.exists() and any(output.iterdir()):
        if not restart:
            raise FileExistsError(f"incomplete or incompatible run: {output}; use --restart to archive it")
        archive = output.with_name(output.name + datetime.now(timezone.utc).strftime("_archived_%Y%m%dT%H%M%S%f"))
        output.rename(archive)
    output.mkdir(parents=True, exist_ok=True)
    configure_torch(config.torch_threads)
    validation_cases = load_cases(validation)
    test_cases = load_cases(test) if test else []
    metadata = {
        "schema_version": 1, "status": "running", "config": config.snapshot(), "seed": seed,
        "fingerprint": fingerprint, "training_source_hash": training_source_hash(),
        "environment": asdict(load_environment(config.environment)),
        "git": git_info(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(), "device": "cpu",
        "dependencies": {name: version(name) for name in
                         ("numpy", "scipy", "gymnasium", "stable-baselines3", "torch", "PyYAML")},
        "validation_sha256": file_hash(validation), "test_sha256": file_hash(test) if test else None,
        "selection": "final checkpoint at the fixed interaction budget",
    }
    write_json(output / "metadata.json", metadata)
    shutil.copyfile(validation, output / "validation_cases.json")
    if test:
        shutil.copyfile(test, output / "test_cases.json")
    normalizer = make_normalizer(config, seed)
    model = ObservedPPO(
        "MlpPolicy", normalizer, device="cpu", seed=seed, verbose=0,
        n_steps=config.n_steps, batch_size=config.batch_size, n_epochs=config.n_epochs,
        learning_rate=config.learning_rate, gamma=config.gamma, gae_lambda=config.gae_lambda,
        clip_range=config.clip_range, clip_range_vf=None, ent_coef=config.ent_coef,
        vf_coef=config.vf_coef, max_grad_norm=config.max_grad_norm,
        normalize_advantage=config.normalize_advantage, target_kl=None,
        policy_kwargs={"net_arch": {"pi": list(config.net_arch), "vf": list(config.net_arch)},
                       "activation_fn": torch.nn.Tanh},
    )
    model.set_logger(configure(str(output), ["csv"]))
    controller = PPOController(model, normalizer)
    curve, validation_episodes = [], []
    evaluation_seconds = 0.0
    started = time.perf_counter()

    def validate(model):
        nonlocal evaluation_seconds
        if model.num_timesteps and model.num_timesteps % config.eval_every and model.num_timesteps != config.total_timesteps:
            return
        for parameter in model.policy.parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError("nonfinite policy parameter")
        begin = time.perf_counter()
        with frozen_normalization(normalizer):
            rows, _ = evaluate_cases(controller, validation_cases, config.environment)
        evaluation_seconds += time.perf_counter() - begin
        metrics = mean_metrics(rows)
        curve.append({"timesteps": model.num_timesteps, **metrics})
        validation_episodes.extend({"timesteps": model.num_timesteps, **row} for row in rows)
        write_csv(output / "validation_curve.csv", curve)
        write_csv(output / "validation_episodes.csv", validation_episodes)
        for key, value in metrics.items():
            model.logger.record(f"validation/{key}", value)
        if model.num_timesteps:
            model.save(output / "latest_model.zip")
            normalizer.save(output / "latest_normalizer.pkl")
        print(f"{config.name} seed={seed} steps={model.num_timesteps} "
              f"validation_return={metrics['return_raw']:.3f} cvr={metrics['cvr_pct']:.2f}% "
              f"kwh={metrics['energy_kwh']:.3f}", flush=True)

    try:
        validate(model)
        model.update_observer = validate
        model.learn(total_timesteps=config.total_timesteps, log_interval=None, callback=ActionDiagnostics())
        learning_wall = time.perf_counter() - started
        if model.num_timesteps != config.total_timesteps:
            raise RuntimeError("actual sample budget differs from the frozen protocol")
        model.save(output / "model.zip")
        normalizer.save(output / "normalizer.pkl")
        probe_env = HVAC2R2CEnv(config.environment)
        try:
            probes = []
            for case in validation_cases[:3]:
                observation, _ = probe_env.reset(seed=case.reset_seed, options=case.options())
                probes.append({"observation": observation.tolist(), "action": controller(observation).tolist()})
            metadata["reload_probes"] = probes
        finally:
            probe_env.close()
        artifact_names = ["model.zip", "normalizer.pkl", "validation_curve.csv", "validation_episodes.csv", "progress.csv"]
        final_test_seconds = 0.0
        if test_cases:
            begin = time.perf_counter()
            with frozen_normalization(normalizer):
                rows, traces = evaluate_cases(controller, test_cases, config.environment, collect_traces=True)
            final_test_seconds = time.perf_counter() - begin
            write_csv(output / "test_episodes.csv", rows)
            representative = [row for row in traces if row["case_id"].endswith("_00")]
            write_csv(output / "test_traces.csv", representative)
            artifact_names += ["test_episodes.csv", "test_traces.csv"]
            metadata["test_mean"] = mean_metrics(rows)
        metadata.update({
            "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "actual_timesteps": model.num_timesteps, "learning_wall_seconds": learning_wall,
            "periodic_validation_seconds": evaluation_seconds,
            "training_seconds_excluding_validation": learning_wall - evaluation_seconds,
            "final_test_seconds": final_test_seconds, "final_validation": curve[-1],
            "parameter_count": sum(parameter.numel() for parameter in model.policy.parameters()),
            "artifacts": {name: file_hash(output / name) for name in artifact_names},
        })
        write_json(output / "metadata.json", metadata)
        print(f"COMPLETE {config.name} seed={seed} wall_seconds={learning_wall:.1f}", flush=True)
        return metadata
    except BaseException:
        metadata.update(status="failed", error=traceback.format_exc())
        write_json(output / "metadata.json", metadata)
        raise
    finally:
        normalizer.close()


def load_controller(run: Path) -> tuple[PPOController, VecNormalize]:
    metadata = read_json(run / "metadata.json")
    config = from_dict(metadata["config"])
    configure_torch(config.torch_threads)
    env = DummyVecEnv([lambda: HVAC2R2CEnv(config.environment)])
    normalizer = VecNormalize.load(run / "normalizer.pkl", env)
    normalizer.training, normalizer.norm_reward = False, False
    model = PPO.load(run / "model.zip", device="cpu")
    return PPOController(model, normalizer), normalizer

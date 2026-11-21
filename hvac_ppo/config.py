"""Validated, serializable experiment settings; all studied values are constants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
import re

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "default"
    total_timesteps: int = 512_000
    eval_every: int = 20_480
    n_envs: int = 1
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: tuple[int, ...] = (64, 64)
    torch_threads: int = 1
    normalize_advantage: bool = True
    normalize_observations: bool = True
    normalize_rewards: bool = False
    clip_obs: float = 10.0
    environment: dict = field(default_factory=dict)

    def __post_init__(self):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", self.name):
            raise ValueError("name must be a safe directory component")
        for name in ("total_timesteps", "eval_every", "n_envs", "n_steps", "batch_size", "n_epochs", "torch_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        rollout_size = self.n_steps * self.n_envs
        if rollout_size <= 1 or self.batch_size <= 1 or rollout_size % self.batch_size:
            raise ValueError("rollout must contain complete mini-batches with at least two samples")
        if self.total_timesteps % rollout_size or self.eval_every % rollout_size:
            raise ValueError("training/evaluation budgets must align with complete rollouts")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1 or not 0 < self.clip_range < 1:
            raise ValueError("invalid discount, GAE, or clipping parameter")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0 or self.clip_obs <= 0:
            raise ValueError("learning rate, gradient norm and observation limit must be positive")
        if self.ent_coef < 0 or self.vf_coef < 0:
            raise ValueError("loss coefficients must be nonnegative")
        if not self.net_arch or any(not isinstance(n, int) or n <= 0 for n in self.net_arch):
            raise ValueError("net_arch must contain positive widths")
        if self.normalize_rewards:
            raise ValueError("this experiment fixes reward normalization off")

    def snapshot(self) -> dict:
        return asdict(self)


def from_dict(values: dict) -> ExperimentConfig:
    unknown = set(values) - {item.name for item in fields(ExperimentConfig)}
    if unknown:
        raise ValueError(f"unknown experiment settings: {sorted(unknown)}")
    values = dict(values)
    if "net_arch" in values:
        values["net_arch"] = tuple(values["net_arch"])
    return ExperimentConfig(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    return from_dict(yaml.safe_load(Path(path).read_text()))


def ablation_configs(base: ExperimentConfig, path: str | Path) -> list[ExperimentConfig]:
    entries = yaml.safe_load(Path(path).read_text())["configurations"]
    result = []
    for entry in entries:
        if set(entry) - {"name", "clip_range", "gae_lambda", "n_epochs"}:
            raise ValueError("ablation changes an uncontrolled setting")
        result.append(replace(base, **entry))
    if len({config.name for config in result}) != len(result):
        raise ValueError("duplicate configuration names")
    return result

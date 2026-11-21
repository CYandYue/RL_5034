"""Validate completed experiments, then export auditable tables and figures."""

from __future__ import annotations

from pathlib import Path
import os

from .io import ROOT, file_hash, read_json, stable_hash, write_json

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ablation_configs, load_config
from .evaluation import METRICS
from .scenarios import load_cases
from .training import complete_run, run_identity

FAMILIES = {
    "Clip range": ["clip_010", "default", "clip_030"],
    "GAE lambda": ["lambda_090", "default", "lambda_099"],
    "Epochs": ["epochs_05", "default", "epochs_20"],
}
LABELS = {"default": "Default", "clip_010": "clip=0.1", "clip_030": "clip=0.3",
          "lambda_090": "lambda=0.90", "lambda_099": "lambda=0.99",
          "epochs_05": "epochs=5", "epochs_20": "epochs=20"}
COLORS = ["#bc6335", "#226c5b", "#435e9c"]


def save_figure(fig, directory: Path, name: str):
    fig.tight_layout()
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def summarize(runs: Path, rules: Path, output: Path):
    import yaml
    base = load_config(ROOT / "configs/ppo/base.yaml")
    configs = ablation_configs(base, ROOT / "configs/ppo/ablation.yaml")
    seeds = yaml.safe_load((ROOT / "configs/ppo/ablation.yaml").read_text())["training_seeds"]
    validation_path, test_path = ROOT / "configs/scenarios/validation.json", ROOT / "configs/scenarios/test.json"
    expected_cases = {case.case_id for case in load_cases(test_path)}
    episodes, curves, diagnostics, runtime, stability, manifests = [], [], [], [], [], []
    environment_hashes, source_hashes = set(), set()
    for config in configs:
        for seed in seeds:
            directory = runs / config.name / f"seed_{seed}"
            if not complete_run(directory, run_identity(config, seed, validation_path, test_path)):
                raise ValueError(f"missing, modified, or incompatible complete run: {directory}")
            metadata = read_json(directory / "metadata.json")
            environment_hashes.add(stable_hash(metadata["environment"]))
            source_hashes.add(metadata["training_source_hash"])
            table = pd.read_csv(directory / "test_episodes.csv")
            if set(table.case_id) != expected_cases or table.case_id.duplicated().any():
                raise ValueError(f"wrong test conditions: {directory}")
            if not np.isfinite(table[list(METRICS)].to_numpy()).all():
                raise ValueError(f"nonfinite final metrics: {directory}")
            if not np.allclose(table.return_raw, -table.degree_hours - 0.1 * table.energy_kwh):
                raise ValueError("reported reward disagrees with physical metrics")
            table.insert(0, "seed", seed)
            table.insert(0, "configuration", config.name)
            episodes.append(table)
            curve = pd.read_csv(directory / "validation_curve.csv")
            if curve.timesteps.tolist() != [0, *range(config.eval_every, config.total_timesteps + 1, config.eval_every)]:
                raise ValueError(f"incomplete validation curve: {directory}")
            curve.insert(0, "seed", seed)
            curve.insert(0, "configuration", config.name)
            curves.append(curve)
            progress = pd.read_csv(directory / "progress.csv")
            progress.insert(0, "seed", seed)
            progress.insert(0, "configuration", config.name)
            diagnostics.append(progress)
            runtime.append({"configuration": config.name, "seed": seed,
                            "training_seconds": metadata["training_seconds_excluding_validation"],
                            "validation_seconds": metadata["periodic_validation_seconds"],
                            "learning_wall_seconds": metadata["learning_wall_seconds"],
                            "final_test_seconds": metadata["final_test_seconds"],
                            "timesteps": metadata["actual_timesteps"],
                            "optimizer_steps": config.total_timesteps // config.batch_size * config.n_epochs})
            late = curve[curve.timesteps >= config.total_timesteps / 2]
            stability.append({"configuration": config.name, "seed": seed,
                              "late_validation_return_std": late.return_raw.std(ddof=1),
                              "late_peak_to_final_drop": late.return_raw.max() - curve.return_raw.iloc[-1],
                              "max_update_kl": progress["train/approx_kl"].max(),
                              "mean_update_kl": progress["train/approx_kl"].mean(),
                              "mean_clip_fraction": progress["train/clip_fraction"].mean(),
                              "final_policy_std": progress["train/std"].iloc[-1]})
            manifests.append({"configuration": config.name, "seed": seed, "fingerprint": metadata["fingerprint"],
                              "git": metadata["git"], "artifacts": metadata["artifacts"]})
    if len(environment_hashes) != 1 or len(source_hashes) != 1:
        raise ValueError("runs do not share the same environment and training implementation")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    all_episodes = pd.concat(episodes, ignore_index=True)
    all_curves = pd.concat(curves, ignore_index=True)
    all_diagnostics = pd.concat(diagnostics, ignore_index=True)
    runtime = pd.DataFrame(runtime)
    stability = pd.DataFrame(stability)
    per_scene = all_episodes.groupby(["configuration", "seed", "scenario"])[list(METRICS)].mean().reset_index()
    overall = all_episodes.groupby(["configuration", "seed"])[list(METRICS)].mean().reset_index()
    overall["scenario"] = "all"
    per_seed = pd.concat([per_scene, overall], ignore_index=True)
    summary = per_seed.groupby(["configuration", "scenario"])[list(METRICS)].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary["n_training_seeds"] = len(seeds)
    for name, table in (("test_episodes", all_episodes), ("validation_curves", all_curves),
                        ("training_diagnostics", all_diagnostics), ("per_seed_metrics", per_seed),
                        ("summary", summary), ("runtime", runtime), ("stability", stability)):
        table.to_csv(output / f"{name}.csv", index=False)
    if read_json(rules / "metadata.json")["case_sha256"] != file_hash(test_path):
        raise ValueError("rule controller metadata refers to different physical test conditions")
    rule_episodes = pd.read_csv(rules / "rule_episodes.csv")
    for controller, table in rule_episodes.groupby("controller"):
        if set(table.case_id) != expected_cases or table.case_id.duplicated().any():
            raise ValueError(f"rule {controller} used different test conditions")
    rule_per_scene = rule_episodes.groupby(["controller", "scenario"])[list(METRICS)].mean().reset_index()
    rule_all = rule_episodes.groupby("controller")[list(METRICS)].mean().reset_index()
    rule_all["scenario"] = "all"
    rule_summary = pd.concat([rule_per_scene, rule_all], ignore_index=True)
    rule_summary.to_csv(output / "rule_summary.csv", index=False)
    baseline = overall[overall.configuration == "default"].set_index("seed")
    paired = []
    for name, table in overall.groupby("configuration"):
        delta = table.set_index("seed")[list(METRICS)] - baseline[list(METRICS)]
        paired.append({"configuration": name, **{f"delta_{key}_{stat}": getattr(delta[key], stat)()
                                                 for key in METRICS for stat in ("mean", "std")}})
    pd.DataFrame(paired).to_csv(output / "paired_differences.csv", index=False)
    write_json(output / "run_manifest.json", {"schema_version": 1, "runs": manifests,
               "test_cases": len(expected_cases), "training_seeds": seeds,
               "training_source_hash": next(iter(source_hashes))})
    facts = {"summary": summary.to_dict("records"), "rules": rule_summary.to_dict("records"),
             "runtime": runtime.groupby("configuration").mean(numeric_only=True).reset_index().to_dict("records"),
             "stability": stability.groupby("configuration").mean(numeric_only=True).reset_index().to_dict("records")}
    write_json(output / "facts.json", facts)
    pd.read_csv(runs / "default/seed_0/test_traces.csv").to_csv(output / "default_seed0_traces.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18, "pdf.fonttype": 42, "ps.fonttype": 42})
    plot_learning(all_curves, figures)
    plot_metrics(summary, rule_summary, figures)
    plot_diagnostics(all_diagnostics, figures)
    plot_runtime(runtime, figures)
    plot_trajectories(runs, rules, figures)
    plot_presentation(curves=all_curves, summary=summary, runs=runs, rules=rules, figures=figures)
    print(f"Validated and summarized {len(manifests)} runs, {len(all_episodes)} test episodes: {output}")


def plot_learning(curves, figures):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8))
    for column, (family, names) in enumerate(FAMILIES.items()):
        for color, name in zip(COLORS, names):
            grouped = curves[curves.configuration == name].groupby("timesteps").return_raw.agg(["mean", "std"])
            for row, start in ((0, 0), (1, 102400)):
                selected = grouped[grouped.index >= start]
                x = selected.index.to_numpy() / 1000
                axes[row, column].plot(x, selected["mean"], color=color, label=LABELS[name])
                axes[row, column].fill_between(x, selected["mean"] - selected["std"],
                                               selected["mean"] + selected["std"], color=color, alpha=.13)
                axes[row, column].set_xlabel("Environment steps (thousands)")
        axes[0, column].set_title(family)
        axes[0, column].legend(fontsize=8)
        axes[1, column].set_title("After 102,400 steps")
    axes[0, 0].set_ylabel("Validation return")
    axes[1, 0].set_ylabel("Validation return")
    save_figure(fig, figures, "learning_curves")


def plot_metrics(summary, rule_summary, figures):
    overall = summary[summary.scenario == "all"].set_index("configuration")
    thermostat = rule_summary[(rule_summary.controller == "thermostat") & (rule_summary.scenario == "all")].iloc[0]
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.5))
    metrics = [("cvr_pct", "Comfort violations (%)"), ("degree_hours", "Discomfort (C h)"),
               ("energy_kwh", "Electricity (kWh / 7 days)")]
    for column, (family, names) in enumerate(FAMILIES.items()):
        for row, (metric, label) in enumerate(metrics):
            selected = overall.loc[names]
            axes[row, column].errorbar(range(3), selected[f"{metric}_mean"],
                                       yerr=selected[f"{metric}_std"], fmt="o-", color="#226c5b", capsize=4)
            axes[row, column].axhline(thermostat[metric], color="#a25d35", linestyle="--", label="Thermostat")
            axes[row, column].set_xticks(range(3), [LABELS[name] for name in names], fontsize=8)
            if column == 0:
                axes[row, column].set_ylabel(label)
            if row == 0:
                axes[row, column].set_title(family)
                axes[row, column].legend(fontsize=8)
    save_figure(fig, figures, "test_metrics")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for index, name in enumerate(overall.index):
        row = overall.loc[name]
        ax.errorbar(row.energy_kwh_mean, row.cvr_pct_mean, xerr=row.energy_kwh_std, yerr=row.cvr_pct_std,
                    fmt="o", capsize=3, label=LABELS[name], color=plt.cm.tab10(index))
    ax.scatter(thermostat.energy_kwh, thermostat.cvr_pct, marker="D", s=60, color="#222222", label="Thermostat")
    ax.set(xlabel="Electricity (kWh / 7 days)", ylabel="Comfort violations (%)")
    ax.legend(fontsize=8, loc="best")
    save_figure(fig, figures, "comfort_energy_tradeoff")


def plot_diagnostics(diagnostics, figures):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for column, (family, names) in enumerate(FAMILIES.items()):
        for color, name in zip(COLORS, names):
            grouped = diagnostics[diagnostics.configuration == name].groupby("time/total_timesteps").mean(numeric_only=True)
            for row, (key, label) in enumerate((("train/approx_kl", "Approximate KL"),
                                               ("train/clip_fraction", "Clip fraction"))):
                smoothed = grouped[key].rolling(5, min_periods=1).mean()
                axes[row, column].plot(grouped.index.to_numpy() / 1000, smoothed, color=color, label=LABELS[name])
                axes[row, column].set_xlabel("Environment steps (thousands)")
                if column == 0:
                    axes[row, column].set_ylabel(label)
        axes[0, column].set_title(family)
        axes[0, column].legend(fontsize=8)
    fig.suptitle("Across-seed means; 5-update moving average", fontsize=11)
    save_figure(fig, figures, "update_diagnostics")


def plot_runtime(runtime, figures):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    grouped = runtime.groupby("configuration").training_seconds.agg(["mean", "std"])
    names = ["default", "clip_010", "clip_030", "lambda_090", "lambda_099", "epochs_05", "epochs_20"]
    selected = grouped.loc[names]
    ax.bar(range(len(names)), selected["mean"], yerr=selected["std"], color="#528473", capsize=4)
    ax.set_xticks(range(len(names)), [LABELS[name] for name in names], fontsize=8)
    ax.set_ylabel("Training seconds (excludes validation)")
    save_figure(fig, figures, "runtime")


def plot_trajectories(runs, rules, figures):
    ppo = pd.read_csv(runs / "default/seed_0/test_traces.csv")
    thermostat = pd.read_csv(rules / "thermostat_traces.csv")
    fig, axes = plt.subplots(4, 3, figsize=(12, 9), sharex=True)
    for column, scenario in enumerate(("cold", "hot", "transition")):
        p = ppo[ppo.scenario == scenario]
        rule = thermostat[thermostat.scenario == scenario]
        for table, label, color in ((p, "PPO default, seed 0", "#226c5b"), (rule, "Thermostat", "#b86c43")):
            hours = table.time_h / 24
            axes[0, column].plot(hours, table.indoor_c, label=label, color=color, linewidth=1.3)
            axes[2, column].step(hours, table.action, where="pre", color=color, linewidth=.7, alpha=.8)
            axes[3, column].plot(hours, table.cumulative_kwh, color=color)
        axes[0, column].fill_between(p.time_h / 24, p.comfort_low_c, p.comfort_high_c, color="#70966c", alpha=.13)
        axes[0, column].set_title(scenario.capitalize())
        axes[0, column].legend(fontsize=7)
        axes[1, column].plot(p.time_h / 24, p.outdoor_c, color="#65758b")
        axes[3, column].set_xlabel("Elapsed days")
    for row, label in enumerate(("Indoor (C)", "Outdoor (C)", "Action", "Cumulative kWh")):
        axes[row, 0].set_ylabel(label)
    save_figure(fig, figures, "representative_trajectories")


def plot_presentation(curves, summary, runs, rules, figures):
    fig, ax = plt.subplots(figsize=(9, 4.3))
    for index, name in enumerate(LABELS):
        grouped = curves[curves.configuration == name].groupby("timesteps").return_raw.mean()
        ax.plot(grouped.index.to_numpy() / 1000, -grouped, label=LABELS[name], color=plt.cm.tab10(index))
    ax.set(xlabel="Environment steps (thousands)", ylabel="Validation cost (-return; log scale)", yscale="log")
    ax.legend(fontsize=9, ncol=2)
    save_figure(fig, figures, "slides_learning")
    overall = summary[summary.scenario == "all"].set_index("configuration")
    for family, names in FAMILIES.items():
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
        for ax, metric, label in zip(axes, ["return_raw", "cvr_pct", "energy_kwh"],
                                      ["Return (higher is better)", "CVR (%)", "Electricity (kWh)"]):
            table = overall.loc[names]
            ax.errorbar(range(3), table[metric + "_mean"], yerr=table[metric + "_std"],
                        fmt="o-", color="#226c5b", capsize=5)
            ax.set_xticks(range(3), [LABELS[name] for name in names], fontsize=9)
            ax.set_title(label, fontsize=11)
        name = {"Clip range": "clip", "GAE lambda": "lambda", "Epochs": "epochs"}[family]
        save_figure(fig, figures, f"slides_{name}")
    ppo = pd.read_csv(runs / "default/seed_0/test_traces.csv")
    rule = pd.read_csv(rules / "thermostat_traces.csv")
    ppo, rule = ppo[ppo.scenario == "cold"], rule[rule.scenario == "cold"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)
    for table, label, color in ((ppo, "Default PPO, seed 0", "#226c5b"), (rule, "Thermostat", "#b86c43")):
        axes[0].plot(table.time_h / 24, table.indoor_c, label=label, color=color)
        axes[1].plot(table.time_h / 24, table.cumulative_kwh, color=color)
    axes[0].fill_between(ppo.time_h / 24, ppo.comfort_low_c, ppo.comfort_high_c, alpha=.13, color="#70966c")
    axes[0].set_ylabel("Indoor (C)")
    axes[0].legend(fontsize=9)
    axes[1].set(xlabel="Elapsed days", ylabel="Cumulative kWh")
    save_figure(fig, figures, "slides_cold_trace")

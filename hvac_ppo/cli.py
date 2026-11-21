"""Command-line entry points for the complete experiment workflow."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys
import time

import yaml

from .config import ablation_configs, load_config
from .evaluation import evaluate_cases, evaluate_rules, frozen_normalization, mean_metrics
from .io import ROOT, file_hash, read_json, write_csv, write_json
from .scenarios import load_cases, save_default_cases


def make_parser():
    parser = argparse.ArgumentParser(description="AIAA 5034 reproducible HVAC PPO experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="create fixed validation/test conditions without overwriting changes")
    prepare.add_argument("--output", type=Path, default=ROOT / "configs/scenarios")

    rules = sub.add_parser("baseline", help="evaluate off and midpoint-deadband thermostat rules")
    rules.add_argument("--cases", type=Path, default=ROOT / "configs/scenarios/validation.json")
    rules.add_argument("--output", type=Path, default=ROOT / "artifacts/baselines-validation")

    train = sub.add_parser("train", help="train one seeded PPO run")
    train.add_argument("--config", type=Path, default=ROOT / "configs/ppo/base.yaml")
    train.add_argument("--ablation", type=Path, default=ROOT / "configs/ppo/ablation.yaml")
    train.add_argument("--variant", default="default")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--timesteps", type=int)
    train.add_argument("--eval-every", type=int)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--validation", type=Path, default=ROOT / "configs/scenarios/validation.json")
    train.add_argument("--test", type=Path, help="omit during pilot training; final test only after protocol freeze")
    train.add_argument("--restart", action="store_true", help="archive an incomplete/incompatible run and start over")

    batch = sub.add_parser("ablate", help="run independent jobs in separate CPU processes; skip verified complete runs")
    batch.add_argument("--config", type=Path, default=ROOT / "configs/ppo/base.yaml")
    batch.add_argument("--ablation", type=Path, default=ROOT / "configs/ppo/ablation.yaml")
    batch.add_argument("--validation", type=Path, default=ROOT / "configs/scenarios/validation.json")
    batch.add_argument("--test", type=Path, default=ROOT / "configs/scenarios/test.json")
    batch.add_argument("--output", type=Path, default=ROOT / "runs/ablation")
    batch.add_argument("--jobs", type=int, default=2)
    batch.add_argument("--restart", action="store_true")

    evaluate = sub.add_parser("evaluate", help="reload a run and evaluate without updating normalization")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--cases", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    status = sub.add_parser("status", help="inspect batch metadata and latest validation steps")
    status.add_argument("--runs", type=Path, default=ROOT / "runs/ablation")

    summary = sub.add_parser("summarize", help="validate artifacts and create result tables/figures")
    summary.add_argument("--runs", type=Path, default=ROOT / "runs/ablation")
    summary.add_argument("--rules", type=Path, default=ROOT / "results/baselines")
    summary.add_argument("--output", type=Path, default=ROOT / "results")
    report = sub.add_parser("report", help="build the ten-page report, presentation, and Chinese results guide")
    report.add_argument("--results", type=Path, default=ROOT / "results")
    return parser


def run_batch(args) -> int:
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    base = load_config(args.config)
    configs = ablation_configs(base, args.ablation)
    seeds = yaml.safe_load(args.ablation.read_text())["training_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("training seeds must be unique nonnegative integers")
    args.output.mkdir(parents=True, exist_ok=True)
    console_dir = args.output / "_console"
    console_dir.mkdir(exist_ok=True)
    manifest = {"schema_version": 1, "status": "running", "jobs": args.jobs,
                "started_utc": datetime.now(timezone.utc).isoformat(), "runs": []}
    started = time.perf_counter()

    def run_one(config, seed):
        output = args.output / config.name / f"seed_{seed}"
        log = console_dir / f"{config.name}_seed_{seed}.log"
        command = [sys.executable, "-u", "-m", "hvac_ppo", "train", "--config", str(args.config.resolve()),
                   "--ablation", str(args.ablation.resolve()), "--variant", config.name, "--seed", str(seed),
                   "--validation", str(args.validation.resolve()), "--test", str(args.test.resolve()),
                   "--output", str(output.resolve())]
        if args.restart:
            command.append("--restart")
        environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           VECLIB_MAXIMUM_THREADS="1", PYTHONUNBUFFERED="1")
        print(f"START {config.name} seed={seed} log={log}", flush=True)
        with log.open("a") as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        entry = {"configuration": config.name, "seed": seed, "returncode": result.returncode,
                 "path": str(output), "console": str(log)}
        if result.returncode:
            print("FAILED " + str(entry) + "\n" + "\n".join(log.read_text().splitlines()[-12:]), flush=True)
        else:
            print(f"DONE {config.name} seed={seed}", flush=True)
        return entry

    write_json(args.output / "batch_manifest.json", manifest)
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run_one, config, seed) for config in configs for seed in seeds]
        for future in as_completed(futures):
            manifest["runs"].append(future.result())
            write_json(args.output / "batch_manifest.json", manifest)
    failed = any(entry["returncode"] for entry in manifest["runs"])
    manifest.update(status="failed" if failed else "complete", wall_seconds=time.perf_counter() - started,
                    completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output / "batch_manifest.json", manifest)
    return 1 if failed else 0


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.command == "prepare":
        save_default_cases(args.output)
        print(f"Fixed conditions: {args.output}")
    elif args.command == "baseline":
        rows = evaluate_rules(load_cases(args.cases), args.output)
        write_json(args.output / "metadata.json", {"case_sha256": file_hash(args.cases),
                   "thermostat": "comfort midpoint +/-0.5 C deadband; full heating/cooling; otherwise off"})
        for name in ("off", "thermostat"):
            print(name, mean_metrics([row for row in rows if row["controller"] == name]))
    elif args.command == "train":
        from .training import train_run
        base = load_config(args.config)
        configs = {item.name: item for item in ablation_configs(base, args.ablation)}
        if args.variant not in configs:
            raise ValueError(f"unknown variant {args.variant}")
        config = configs[args.variant]
        if args.timesteps is not None:
            config = replace(config, total_timesteps=args.timesteps)
        if args.eval_every is not None:
            config = replace(config, eval_every=args.eval_every)
        train_run(config, args.seed, args.output, args.validation, args.test, args.restart)
    elif args.command == "ablate":
        return run_batch(args)
    elif args.command == "evaluate":
        from .training import load_controller
        controller, normalizer = load_controller(args.run)
        try:
            metadata = read_json(args.run / "metadata.json")
            with frozen_normalization(normalizer):
                rows, traces = evaluate_cases(controller, load_cases(args.cases),
                                             metadata["config"]["environment"], collect_traces=True)
            write_csv(args.output / "episodes.csv", rows)
            write_csv(args.output / "traces.csv", traces)
            write_json(args.output / "summary.json", {"case_sha256": file_hash(args.cases), **mean_metrics(rows)})
            print(mean_metrics(rows))
        finally:
            normalizer.close()
    elif args.command == "status":
        for path in sorted(args.runs.glob("*/seed_*/metadata.json")):
            metadata = read_json(path)
            curve = path.parent / "validation_curve.csv"
            last = curve.read_text().splitlines()[-1].split(",")[0] if curve.exists() else "0"
            print(path.parent.relative_to(args.runs), metadata["status"], "steps=" + last)
    elif args.command == "summarize":
        from .analysis import summarize
        summarize(args.runs, args.rules, args.output)
    elif args.command == "report":
        from .reporting import build_deliverables
        build_deliverables(args.results)
    return 0

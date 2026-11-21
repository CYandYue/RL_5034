"""Small, explicit file formats for reproducible experiment artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_source_hash() -> str:
    paths = [ROOT / "hvac_ppo" / name for name in
             ("config.py", "io.py", "scenarios.py", "evaluation.py", "training.py")]
    paths += sorted((ROOT / "RL_assignment/hvac_env").rglob("*.py"))
    paths += [ROOT / "RL_assignment/hvac_env/configs/default.yaml"]
    return stable_hash({str(p.relative_to(ROOT)): file_hash(p) for p in paths if p.exists()})


def git_info() -> dict:
    def run(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}

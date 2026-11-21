"""Fixed physical conditions, independent of training seeds and policy RNGs."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from hvac_env.config import load_config
from hvac_env.weather import SCENARIOS, outdoor_temperature

from .io import read_json, write_json


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    scenario: str
    weather_delta_c: float
    indoor_init_c: float
    envelope_init_c: float
    reset_seed: int

    def options(self):
        return {key: value for key, value in asdict(self).items()
                if key not in {"case_id", "reset_seed"}}


def generate_cases(split: str, seed: int, per_scenario: int) -> dict:
    if split not in {"validation", "test"} or per_scenario < 1:
        raise ValueError("invalid split or sample count")
    config = load_config()
    rng = np.random.default_rng(seed)
    cases = []
    for scenario_name in SCENARIOS:
        scenario = config.weather.scenario(scenario_name)
        midpoint = (scenario.comfort_low_c + scenario.comfort_high_c) / 2
        for index in range(per_scenario):
            delta = float(rng.uniform(-2, 2))
            indoor = midpoint + float(rng.uniform(-2, 2))
            if split == "test" and index == 0:
                delta, indoor = 0.0, midpoint
            outdoor = outdoor_temperature(0, scenario, delta, config.weather.phase_h)
            envelope = config.thermal.equilibrium_envelope_c(indoor, outdoor)
            case = EvalCase(f"{split}_{scenario_name}_{index:02d}", scenario_name, delta,
                            indoor, envelope, int(rng.integers(0, 2**31 - 1)))
            cases.append(asdict(case))
    return {"schema_version": 1, "split": split, "generation_seed": seed,
            "per_scenario": per_scenario, "cases": cases}


def load_cases(path: str | Path) -> list[EvalCase]:
    data = read_json(path)
    if data["schema_version"] != 1:
        raise ValueError("unsupported scenario schema")
    result = [EvalCase(**case) for case in data["cases"]]
    if not result or len({case.case_id for case in result}) != len(result):
        raise ValueError("scenario table must contain unique cases")
    for case in result:
        if case.scenario not in SCENARIOS or not np.isfinite(
            [case.weather_delta_c, case.indoor_init_c, case.envelope_init_c]
        ).all():
            raise ValueError(f"invalid condition: {case.case_id}")
    return result


def save_default_cases(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for split, seed, count in (("validation", 1001, 5), ("test", 2001, 10)):
        path = directory / f"{split}.json"
        if path.exists():
            if read_json(path) != generate_cases(split, seed, count):
                raise ValueError(f"refusing to overwrite modified conditions: {path}")
        else:
            write_json(path, generate_cases(split, seed, count))

"""Export fixed-input temperature and electricity traces using the public Gym API."""

import argparse
import csv
from pathlib import Path

import numpy as np

from hvac_env import HVAC2R2CEnv
from hvac_env.weather import SCENARIOS


def fixed_trace(scenario: str, action: float, duration_h: float) -> list[dict]:
    env = HVAC2R2CEnv({"episode_days": duration_h / 24})
    command = np.array([action], dtype=np.float32)
    _, initial = env.reset(seed=42, options={
        "scenario": scenario, "weather_delta_c": 0, "indoor_init_c": 22, "envelope_init_c": 18,
    })
    envelope_before = initial["envelope_temp_c"]
    rows = []
    for _ in range(env.config.episode_steps):
        obs, _, _, _, info = env.step(command)
        rows.append({
            "scenario": scenario,
            "step_index": info["step_index"],
            "time_h_start": info["time_h_start"],
            "time_h": info["time_h_end"],
            "T_i_start_c": info["indoor_temp_c_before"],
            "T_e_start_c": envelope_before,
            "T_i_c": info["indoor_temp_c"],
            "T_e_c": info["envelope_temp_c"],
            "T_o_c": float(obs[2]),
            "T_o_applied_c": info["outdoor_temp_c_applied"],
            "Q_g_applied_kw": info["internal_gain_kw_applied"],
            "u": info["action_applied"],
            "Q_hvac_kw_thermal": info["q_hvac_kw_thermal"],
            "P_el_kw": info["p_electric_kw"],
            "energy_kwh_step": info["energy_kwh_step"],
        })
        envelope_before = info["envelope_temp_c"]
    env.close()
    return rows


def plot_traces(rows: list[dict], output_path: Path) -> None:
    # Keep Matplotlib optional for headless runs that only need CSV data.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    colors = ("#12615e", "#bc5a20", "#506987")
    for ax, name in zip(axes.flat, SCENARIOS):
        trace = [r for r in rows if r["scenario"] == name and r["u"] == 0]
        hours = [0] + [r["time_h"] for r in trace]
        series = (("T_i_c", "T_i_start_c", "Indoor"),
                  ("T_e_c", "T_e_start_c", "Envelope"),
                  ("T_o_c", "T_o_applied_c", "Outdoor"))
        for (field, start, label), color in zip(series, colors):
            temperatures = [trace[0][start]] + [r[field] for r in trace]
            ax.plot(hours, temperatures, color=color, label=label, linewidth=2)
        ax.set_title(f"{name.capitalize()} weather, u = 0")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.legend(frameon=False, fontsize=9)

    ax = axes.flat[3]
    for action, color in zip((-0.5, 0, 0.5), ("#42689a", "#525657", "#af4c36")):
        trace = [r for r in rows if r["scenario"] == "transition" and r["u"] == action and r["time_h"] <= 2]
        hours = [0] + [r["time_h"] for r in trace]
        indoor_c = [trace[0]["T_i_start_c"]] + [r["T_i_c"] for r in trace]
        ax.plot(hours, indoor_c, label=f"u = {action:g}", color=color, linewidth=2)
    ax.set_title("Transition weather, indoor response")
    ax.set_xlim(0, 2)
    ax.legend(frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.set_xlabel("Elapsed time (h)")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.text(0.5, 0.015, "fixed-action physics sanity check; not an optimized controller or PPO result",
             ha="center", fontsize=10)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--plot", action="store_true", help="also write thermal_sanity.png")
    args = parser.parse_args()
    rows = []
    for scenario in SCENARIOS:
        rows.extend(fixed_trace(scenario, action=0.0, duration_h=24))
    for action in (-0.5, 0.5):
        rows.extend(fixed_trace("transition", action=action, duration_h=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "dynamics_trace.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{csv_path}: {len(rows)} intervals")
    if args.plot:
        image_path = args.output_dir / "thermal_sanity.png"
        plot_traces(rows, image_path)
        print(image_path)


if __name__ == "__main__":
    main()

# Experiment results

These files come from the frozen seven-configuration, three-seed protocol. Each model receives 512,000 training interactions and is tested at its final checkpoint on the same 30 conditions. There are 21 trained models and 630 PPO test episodes, plus 60 rule-controller episodes. Conditions are balanced across cold, hot, and transition weather.

## Tables and provenance

| File | Meaning |
| --- | --- |
| `summary.csv` | Mean and sample SD across **three training seeds**, for each weather class and the equally weighted overall result (`scenario=all`). |
| `per_seed_metrics.csv` | Physical-condition averages within each trained model, before aggregating seeds. |
| `test_episodes.csv` | All 630 PPO test episodes; physical metrics in original units. |
| `rule_summary.csv`, `baselines/rule_episodes.csv` | Off and thermostat references on the identical final test table. |
| `validation_curves.csv` | Initial and post-update validation every 20,480 interactions, for all runs. |
| `training_diagnostics.csv` | All 5,250 PPO updates, including KL, clip fraction, value loss, policy SD and action clipping. SB3's `train/n_updates` counts epochs, not mini-batch optimizer steps. |
| `runtime.csv` | Wall time, periodic validation time and actual optimizer counts. Training time subtracts periodic evaluation, but includes normal training bookkeeping. |
| `stability.csv` | Per-model late-validation variability, late peak-to-final drop and update diagnostics. |
| `paired_differences.csv` | Per-seed differences from the default, then mean/SD across paired seed labels; **not** significance tests. |
| `run_manifest.json` | Training-source hash, per-run fingerprints, Git provenance and artifact checksums. |
| `facts.json` | Machine-readable means used for interpretation. |
| `condition_coverage.csv` | Actual offset/initial-temperature coverage by weather and split. The small validation realization has only negative offsets in hot and transition weather; validation rankings need not match final-test rankings. |
| `final_verification.json`, `layout_verification.json` | Original experiment acceptance checks and the later directory-reorganization audit, including preserved data/model hashes, rebuilt document content and local links. |
| `default_seed0_traces.csv`, `baselines/*_traces.csv` | Preselected condition 00 for each weather class; initial point plus 672 completed intervals. |
| `pilot/` | Default validation-only pilot used to establish the formal budget before final tests. |

CVR is the percentage of completed interval endpoints outside the comfort band (tolerance 1e−8 °C). Degree-hours integrate deviation magnitude; kWh integrates actuator electricity. Raw return obeys `return = −degree_hours − 0.1 × kWh`. A low CVR and a low degree-hour value measure different properties. All energy values refer to **one seven-day episode**, averaged over conditions, rather than the total electricity across test episodes.

Thirty test episodes are repeated conditions for one learned model, not thirty independent training repetitions. Error bars in the figures are across-training-seed sample SD, not confidence intervals. Seed labels are paired across configurations, while each run starts a fresh training process. Three seeds support descriptive comparisons only.

## Figures and rebuilding

`figures/` contains both vector PDF and PNG versions of the learning curves, physical metrics, comfort–energy tradeoff, update diagnostics, runtime, and preselected trajectories. The `slides_*` figures use the same data with layouts for presentation. Learning bands show across-seed SD; update-diagnostic lines are seed means followed by a five-update moving average.

For editing presentations, use the portable [figure asset folder and offline gallery](../deliverables/figure_assets/index.html). It contains verified PNG/PDF copies and a Chinese chart guide. Refresh it after regenerating figures with `python scripts/export_figure_assets.py`; the report continues to use `results/figures/`.

With complete local models and logs present:

```bash
conda activate ppo
python -m hvac_ppo summarize
```

The summarizer verifies all run fingerprints and saved-artifact hashes before accepting them. Model archives and detailed per-run files remain in the Git-ignored `runs/ablation/` directory. Rebuild the English report, slides and offline Chinese guide from the committed result tables and figures with:

```bash
python -m hvac_ppo report
```

This document build requires `pdflatex` and common TeX Live packages; PPO training itself does not. See [the Chinese results guide](../docs/results/RESULTS_zh.html) and [the complete protocol](../docs/experiments/EXPERIMENT_PROTOCOL.md).

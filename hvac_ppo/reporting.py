"""Build the ten-page report, presentation, and Chinese reading guide from results."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import html
from pathlib import Path
import re
import shutil
import subprocess

import pandas as pd
from pypdf import PdfReader

from .analysis import FAMILIES, LABELS
from .io import ROOT, file_hash, read_json, write_json


def tex_escape(text):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return re.sub(r"[\\&%$#_{}]", lambda match: replacements[match.group()], str(text))


def pm(row, metric, digits=3, math=True):
    value = f"{row[metric + '_mean']:.{digits}f} \\pm {row[metric + '_std']:.{digits}f}"
    return r"\ensuremath{" + value + "}" if math else value


def compile_pdf(directory: Path, source: str, destination: str, expected_pages: int):
    build = directory / "build"
    build.mkdir(exist_ok=True)
    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", source]
    for _ in range(2):
        result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"LaTeX failed for {source}:\n" + result.stdout[-7000:])
    pdf = build / Path(source).with_suffix(".pdf").name
    reader = PdfReader(pdf)
    if len(reader.pages) != expected_pages:
        raise ValueError(f"{source}: expected {expected_pages} pages, got {len(reader.pages)}")
    if any(len(page.extract_text().strip()) < 100 for page in reader.pages):
        raise ValueError(f"blank or almost-empty page in {source}")
    shutil.copyfile(pdf, directory / destination)
    return {"pages": len(reader.pages), "sha256": file_hash(directory / destination)}


def build_deliverables(results: Path):
    if results.resolve() != (ROOT / "results").resolve():
        raise ValueError("the course document templates use the canonical repository results/ directory")
    summary = pd.read_csv(results / "summary.csv")
    overall = summary[summary.scenario == "all"].set_index("configuration")
    rules = pd.read_csv(results / "rule_summary.csv")
    thermostat = rules[(rules.controller == "thermostat") & (rules.scenario == "all")].iloc[0]
    default = overall.loc["default"]
    best_name = overall.return_raw_mean.idxmax()
    best = overall.loc[best_name]
    runtime = pd.read_csv(results / "runtime.csv").groupby("configuration").mean(numeric_only=True)
    stability = pd.read_csv(results / "stability.csv").groupby("configuration").mean(numeric_only=True)
    curves = pd.read_csv(results / "validation_curves.csv")
    default_curve = curves[curves.configuration == "default"].groupby("timesteps").mean(numeric_only=True)
    energy_change = 100 * (default.energy_kwh_mean / thermostat.energy_kwh - 1)
    cvr_change = default.cvr_pct_mean - thermostat.cvr_pct
    change_words = f"{abs(energy_change):.1f} percent {'more' if energy_change > 0 else 'less'} electricity"
    worst_late = stability.late_peak_to_final_drop.idxmax()

    def returns(names):
        return ", ".join(f"{overall.loc[name].return_raw_mean:.3f}" for name in names)

    def best_family(names):
        return max(names, key=lambda name: overall.loc[name].return_raw_mean)

    findings = {
        "AbstractFinding": f"The highest observed mean PPO return is {best.return_raw_mean:.3f} for {LABELS[best_name]}; rankings are descriptive given only three training seeds.",
        "LearningFinding": f"Across the default runs, mean validation return changes from {default_curve.return_raw.iloc[0]:.3f} before training to {default_curve.return_raw.iloc[-1]:.3f} at the final budget. The strongest mean validation point is {default_curve.return_raw.max():.3f}; its difference from the final point is retained by the evaluation protocol. Thus a longer budget does not justify assuming that every run's final policy is its best policy.",
        "PerformanceFinding": f"The default uses {change_words} than the thermostat reference. Its CVR is {abs(cvr_change):.3f} percentage points {'higher' if cvr_change > 0 else 'lower'}. Among PPO configurations, {LABELS[best_name]} has the highest observed mean return ({best.return_raw_mean:.3f}), with {best.cvr_pct_mean:.3f} percent CVR and {best.energy_kwh_mean:.3f} kWh. This is an observed grid ranking, not a test-based choice used to retrain or select checkpoints.",
        "ClipFinding": f"Mean final-test returns for clip ranges 0.1, 0.2, and 0.3 are {returns(FAMILIES['Clip range'])}, respectively. Corresponding mean update KL estimates are " + ", ".join(f"{stability.loc[name].mean_update_kl:.4g}" for name in FAMILIES["Clip range"]) + ". The best mean in this slice is " + LABELS[best_family(FAMILIES["Clip range"])] + ". These diagnostics describe realized updates and do not establish a hard trust-region constraint.",
        "LambdaFinding": f"At lambda 0.90, 0.95, and 0.99, mean final-test returns are {returns(FAMILIES['GAE lambda'])}. Across-seed return standard deviations are " + ", ".join(f"{overall.loc[name].return_raw_std:.3f}" for name in FAMILIES["GAE lambda"]) + ". At fixed gamma and architecture, this comparison changes the advantage estimator. Its observed ranking should not be extrapolated to every thermal timescale or value-function accuracy.",
        "EpochFinding": f"Five, ten, and twenty epochs give final-test returns of {returns(FAMILIES['Epochs'])}. The corresponding optimization counts are 40,000, 80,000, and 160,000 mini-batch steps. " + LABELS[best_family(FAMILIES["Epochs"])] + " has the highest observed mean return within this slice; additional optimization is not intrinsically a guarantee of better closed-loop control.",
        "StabilityFinding": f"The default's mean late-validation standard deviation is {stability.loc['default'].late_validation_return_std:.3f}. The largest mean late peak-to-final drop occurs for {LABELS[worst_late]} ({stability.loc[worst_late].late_peak_to_final_drop:.3f} return units).",
        "RuntimeFinding": "Mean training wall times excluding periodic validation are " + ", ".join(f"{runtime.loc[name].training_seconds:.1f} seconds" for name in FAMILIES["Epochs"]) + " for 5, 10, and 20 epochs, respectively.",
        "ConclusionFinding": f"Within the frozen seven-configuration experiment, {LABELS[best_name]} achieves the highest observed mean final-test return. The default's physical outcomes are {default.cvr_pct_mean:.3f} percent CVR, {default.degree_hours_mean:.3f} degree-hours, and {default.energy_kwh_mean:.3f} kWh per seven-day episode. Relative to the thermostat, it uses {change_words}; the comfort comparison must be considered separately. The principal result is a task-specific account of sample efficiency, comfort--energy balance, update variability, and computational cost, rather than a universal optimal PPO setting.",
    }
    traces = pd.read_csv(results / "default_seed0_traces.csv")
    medians = [traces[(traces.scenario == name) & (traces.step > 0)].indoor_c.median()
               for name in ("cold", "hot", "transition")]
    findings["TrajectoryFinding"] = "For these fixed cold, hot, and transition examples, the default PPO's median indoor temperatures are " + ", ".join(f"{value:.2f}" for value in medians) + " degrees C, respectively. The comfort bands and thermostat setpoints supply context for where the learned controller operates. Cumulative electricity is monotone and ends at the episode energy; temperature boundary crossings and action changes should be interpreted jointly rather than equated with instability on their own."
    findings.update({
        "SlidePerformance": f"Default: {default.cvr_pct_mean:.2f} percent CVR, {default.energy_kwh_mean:.2f} kWh; thermostat: {thermostat.cvr_pct:.2f} percent, {thermostat.energy_kwh:.2f} kWh.",
        "SlideClip": f"Mean return, small/default/large: {returns(FAMILIES['Clip range'])}.",
        "SlideLambda": f"Mean return, small/default/large: {returns(FAMILIES['GAE lambda'])}.",
        "SlideEpochs": f"Mean training time, 5/10/20 epochs: {runtime.loc['epochs_05'].training_seconds:.0f}/{runtime.loc['default'].training_seconds:.0f}/{runtime.loc['epochs_20'].training_seconds:.0f} seconds.",
        "SlideStability": f"Largest mean late-validation peak-to-final drop: {LABELS[worst_late]}, {stability.loc[worst_late].late_peak_to_final_drop:.2f} return units.",
        "SlideConclusion": f"Highest observed mean PPO return: {LABELS[best_name]} ({best.return_raw_mean:.3f}); this is a descriptive grid result.",
    })
    report = ROOT / "report"
    generated = report / "generated"
    generated.mkdir(exist_ok=True)
    macros = {"DefaultReturn": pm(default, "return_raw"), "DefaultCVR": pm(default, "cvr_pct"),
              "DefaultEnergy": pm(default, "energy_kwh"), "ThermostatEnergy": f"{thermostat.energy_kwh:.3f}",
              "ThermostatCVR": f"{thermostat.cvr_pct:.3f}"}
    lines = [f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items()]
    lines += [f"\\newcommand{{\\{key}}}{{{tex_escape(value)}}}" for key, value in findings.items()]
    (generated / "macros.tex").write_text("% Generated from audited results; do not edit numeric values manually.\n" + "\n".join(lines) + "\n")
    table = [r"\begin{tabular}{lrrrr}", r"\toprule Controller & Return & CVR (\%) & Degree-hours & kWh \\", r"\midrule"]
    slide_table = [r"\begin{tabular}{lrrr}", r"\toprule Controller & Return & CVR (\%) & kWh \\", r"\midrule"]
    for controller in ("off", "thermostat"):
        row = rules[(rules.scenario == "all") & (rules.controller == controller)].iloc[0]
        label = "HVAC off" if controller == "off" else "Thermostat"
        table.append(f"{label} & {row.return_raw:.3f} & {row.cvr_pct:.3f} & {row.degree_hours:.3f} & {row.energy_kwh:.3f} \\\\")
        slide_table.append(f"{label} & {row.return_raw:.2f} & {row.cvr_pct:.2f} & {row.energy_kwh:.2f} \\\\")
    table.append(r"\midrule")
    slide_table.append(r"\midrule")
    for name in LABELS:
        row = overall.loc[name]
        table.append(tex_escape(LABELS[name]) + " & " + " & ".join(pm(row, metric) for metric in
                     ("return_raw", "cvr_pct", "degree_hours", "energy_kwh")) + r" \\")
        slide_table.append(tex_escape(LABELS[name]) + " & " + " & ".join(pm(row, metric, 2) for metric in
                           ("return_raw", "cvr_pct", "energy_kwh")) + r" \\")
    table += [r"\bottomrule", r"\end{tabular}"]
    slide_table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "results_table.tex").write_text("{\\small\n" + "\n".join(table) + "\n}\n")
    (generated / "slides_table.tex").write_text("{\\scriptsize\n" + "\n".join(slide_table) + "\n}\n")
    write_json(generated / "findings.json", findings)
    build_chinese_guide(results, overall, thermostat, findings)
    build_speaker_notes(findings, default, thermostat)
    report_pdf = compile_pdf(report, "final_report.tex", "final_report.pdf", 10)
    slides_pdf = compile_pdf(ROOT / "presentation", "slides.tex", "slides.pdf", 12)
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(), "report": report_pdf,
                "slides": slides_pdf, "authors": "pending team-provided names and IDs",
                "result_sha256": {name: file_hash(results / name) for name in
                                  ("summary.csv", "validation_curves.csv", "runtime.csv", "stability.csv")}}
    write_json(report / "build_manifest.json", manifest)
    print("Built ten-page report, twelve-slide presentation, speaker notes, Q&A, and Chinese HTML guide.")


def build_chinese_guide(results, overall, thermostat, findings):
    rows = []
    for name in LABELS:
        row = overall.loc[name]
        rows.append({"配置": LABELS[name], **{label: f"{row[key + '_mean']:.3f} ± {row[key + '_std']:.3f}"
                     for key, label in (("return_raw", "原始回报"), ("cvr_pct", "CVR (%)"),
                                        ("degree_hours", "度时 (°C·h)"), ("energy_kwh", "电量 (kWh)"))}})
    table = pd.DataFrame(rows).to_html(index=False, border=0)
    best = overall.return_raw_mean.idxmax()
    default = overall.loc["default"]
    paragraphs = [
        "正式实验已按冻结协议完成：7 个单因素配置、每个 3 个训练种子，每次 512,000 步，合计 10,752,000 个训练交互。验证与最终测试条件分离；比较的是末次模型，未按测试集选择检查点。",
        f"默认 PPO 的三种子平均 CVR 为 {default.cvr_pct_mean:.3f}%，电量为 {default.energy_kwh_mean:.3f} kWh/七天；恒温器对应为 {thermostat.cvr_pct:.3f}% 和 {thermostat.energy_kwh:.3f} kWh。比较时必须同时看舒适度和能耗，不能把总回报更高直接等同于两者都更好。",
        f"在已测试的 PPO 配置中，{LABELS[best]} 的平均最终回报最高，为 {overall.loc[best].return_raw_mean:.3f}。这是三种子、小范围网格中的描述性结果，不是所有任务的最优参数，也不代表统计显著性。",
        "表中先对每个训练模型的 30 条固定测试条件取平均，再跨三个训练种子求均值与样本标准差。测试回合数与独立训练次数不同。关闭空调的零能耗不代表合理温控；恒温器是舒适中点 ±0.5°C 死区的满功率参考。",
        "完整曲线显示先消除大幅温度越界、再改变能耗和残余越界的学习过程。后期仍可能出现回落，所以报告保留固定末次模型，并另列后半程波动及最佳点到末点的下降幅度。",
        "clip_range 改变 PPO 目标对概率比变化的裁剪；gae_lambda 改变优势估计；n_epochs 改变同批数据的重复优化次数。KL、裁剪比例、最终物理指标与跨种子波动应一起读，单个诊断值不能独立证明稳定。",
        "这些实验只覆盖合成房间、周期天气、固定 COP、完整热状态可见和三种训练种子。不能据此宣称真实建筑节能比例，或推断参数之间所有交互。英文报告包含完整方法、参考文献及限制。",
    ]
    figures = [("learning_curves", "验证学习曲线：阴影为三个训练种子的样本标准差，下排放大后期。"),
               ("test_metrics", "单因素比较：CVR、度时与电量，误差条为跨训练种子标准差。"),
               ("update_diagnostics", "更新诊断：跨种子均值后做五次更新移动平均，原始值另存 CSV。"),
               ("representative_trajectories", "事先指定的三类天气测试条件 00，默认 PPO 使用种子 0。"),
               ("runtime", "训练耗时排除周期验证；并行负载及 Mac 温度会影响墙钟时间。")]
    images = []
    for name, caption in figures:
        data = base64.b64encode((results / "figures" / f"{name}.png").read_bytes()).decode()
        images.append(f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{data}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    body = "\n".join(f"<p>{html.escape(text)}</p>" for text in paragraphs)
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PPO 实验结果解读</title><style>
body{{margin:0;background:#f1f4ef;color:#253b31;font:17px/1.9 system-ui,"PingFang SC",sans-serif}}main{{max-width:1120px;margin:30px auto;padding:35px 45px;background:#fffefa}}h1,h2{{color:#20583f}}.table{{overflow:auto}}table{{border-collapse:collapse;min-width:850px;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #dce4dd;text-align:left}}th{{background:#eef3ec}}figure{{margin:32px 0}}img{{max-width:100%;height:auto}}figcaption{{color:#65776a;font-size:14px}}a{{color:#216858}}@media(max-width:700px){{main{{margin:0;padding:24px 18px}}}}@media print{{body,main{{background:white;margin:0;padding:0}}figure{{break-inside:avoid}}}}
</style></head><body><main><h1>PPO 温控实验：结果解读</h1>{body}<h2>最终测试指标</h2><div class="table">{table}</div><h2>图表</h2>{''.join(images)}<h2>交付与复现</h2><p><a href="../report/final_report.pdf">10 页英文报告</a> · <a href="../presentation/slides.pdf">12 页展示</a> · <a href="../presentation/speaker_notes.md">英文讲稿</a> · <a href="../presentation/QA_zh.md">中文问答准备</a> · <a href="../results/summary.csv">原始汇总表</a></p><p>完整代码、固定条件、锁定依赖与运行入口位于项目仓库。署名待三位组员补充。</p></main></body></html>'''
    (ROOT / "docs/RESULTS_zh.html").write_text(document)


def build_speaker_notes(findings, default, thermostat):
    sections = [
        ("1. Question and scope (0:00–0:45)", "Our project studies PPO hyperparameters for single-room HVAC control. The goal is to understand how clip range, GAE lambda, and update epochs affect comfort, electricity, and training stability. A teammate supplied the thermal simulator and physical validation. We added the training and evaluation pipeline, ran controlled experiments locally, and analyzed the results. We focus on this synthetic task and do not claim measured savings in a real building."),
        ("2. Physical environment (0:45–1:45)", "The environment has two equivalent temperatures: the indoor node and the envelope node. Thermal resistances determine heat flow, while capacitances store energy and create inertia. The controller sees eleven numbers, including both temperatures, weather, gains, comfort bounds, time, and scenario. Its action controls heating or cooling power every fifteen minutes. The reward penalizes discomfort and electricity. We also report the physical components separately, because a better weighted reward may trade a little comfort for lower energy."),
        ("3. PPO mechanism (1:45–2:50)", "PPO first collects a rollout with the current policy. The critic and GAE estimate whether sampled actions were better or worse than the policy's usual outcome in those states. The actor then changes their likelihood through the clipped objective. Lambda controls the advantage estimator. Clip range changes the surrogate objective. Epochs determine how often we reuse a rollout. The actor and critic have separate small networks. Gradients pass through action probabilities, not through the physical room equations."),
        ("4. Experimental protocol (2:50–3:50)", "We varied one factor at a time around the common default. Sharing the default gives seven configurations, and three seeds each give twenty-one runs. Every run receives exactly five hundred and twelve thousand environment interactions. Architecture, reward, learning rate, gamma, normalization, and sampling settings are fixed. Validation and final-test conditions are separate. We test each run's final checkpoint instead of choosing its best test result. The rules use the same physical conditions, and all experiments run on an M4 CPU under screen."),
        ("5. Learning curves (3:50–4:50)", "This slide shows the validation cost, which is minus return, on a logarithmic scale. Lower is better. The large early penalties reflect poor temperature regulation. Later differences concern electricity and residual discomfort, as well as possible policy deterioration. The full report includes ordinary return plots and across-seed bands. A lower training loss is not a guarantee of monotonic closed-loop improvement, so the fixed final-checkpoint rule is important."),
        ("6. Final metrics (4:50–6:00)", "The table separates return, comfort violation rate, and electricity. " + findings["SlidePerformance"] + " Each PPO entry first averages its thirty test conditions, then averages across three training seeds. The error bars describe training variability; thirty episodes do not provide thirty independent trained models. HVAC off has zero electricity but can have severe discomfort. The thermostat is a meaningful conventional reference. We avoid calling one method universally better when comfort and electricity move in different directions."),
        ("7. Clip range (6:00–6:55)", "Here we compare the three clip ranges while keeping lambda and epochs fixed. " + findings["SlideClip"] + " Clipping discourages additional gains from excessive probability-ratio changes in the preferred direction. It does not impose a hard bound on the final ratio or KL divergence. We therefore examine approximate KL, clipped fraction, learning curves, and final physical outcomes together. The ranking is specific to this experiment and comes from only three seeds."),
        ("8. GAE lambda (6:55–7:50)", "Here gamma stays fixed while lambda changes how TD residuals are combined. " + findings["SlideLambda"] + " Larger lambda uses longer reward fragments, which can reduce dependence on short-horizon value bootstrapping but increase variability. Thermal inertia makes delayed effects relevant, yet it does not imply that the largest lambda must win. The comparison also depends on the value network and rollout settings, so we describe the observed results without a universal monotonic claim."),
        ("9. Epochs and computation (7:50–8:50)", "Epochs change how many times a rollout is optimized. At our common sample budget, five, ten, and twenty epochs correspond to forty thousand, eighty thousand, and one hundred and sixty thousand mini-batch updates. " + findings["SlideEpochs"] + " We compare sample efficiency at equal interactions and report wall time separately. More computation can improve data reuse but can also drive stronger policy changes. The runtime numbers are practical local measurements, affected by parallel load and machine temperature."),
        ("10. Closed-loop trace (8:50–9:50)", "This is the preselected cold-weather test condition zero. The default PPO uses seed zero, and the thermostat starts from identical conditions. The comfort band provides a reference for indoor temperature. The lower plot shows cumulative electricity, not instantaneous power. Different locations within the comfort band can lead to different energy use. The report also shows outdoor forcing, executed actions, and the hot and transition scenes. This is an illustration; aggregate conclusions use every fixed test condition and training seed."),
        ("11. Stability and limitations (9:50–10:55)", "We retain late validation deterioration because we test the final checkpoint. " + findings["SlideStability"] + " The diagnostics are complementary rather than standalone failure criteria. Three seeds limit our uncertainty estimate, and a one-factor design does not resolve every interaction. The simulator also uses synthetic repeating weather, constant COP, and direct observation of envelope temperature. It lacks humidity, equipment cycling, and real-building calibration. These boundaries matter when interpreting the results."),
        ("12. Conclusion (10:55–12:00)", findings["SlideConclusion"] + " Our main outcome is a reproducible account of how these parameters affect learning, comfort, energy, stability, and compute in one controlled task. We do not collapse all these dimensions into a single universal recommendation. The complete workflow runs on a local Mac and includes tests, fixed scenarios, models, logs, and generated figures. Useful extensions would add more seeds, study a limited set of interactions, or introduce calibrated dynamics and realistic weather. Thank you, and we welcome questions."),
    ]
    text = "# English presentation notes\n\nTarget: about 12 minutes including pointing at figures and transitions. Suggested team split: slides 1–4, 5–8, 9–12. Rehearse against the actual PDF; timing is a guide.\n\n"
    text += "\n\n".join(f"## {title}\n\n{paragraph}" for title, paragraph in sections) + "\n"
    (ROOT / "presentation/speaker_notes.md").write_text(text)
    qa = '''# 展示问答准备

- **为什么用 PPO？** 题目指定研究 PPO 的三个参数；低维连续动作也适合使用其高斯策略。我们不宣称它是所有 HVAC 任务中最优算法。
- **为什么只有七个配置？** 默认配置共享，每次只改变一个参数，1+2+2+2=7。该设计不能回答全部交互作用。
- **怎么保证公平？** 相同环境、奖励、网络、采样步数、测试条件和模型选择规则；不同 epochs 的计算成本另外报告。
- **为什么还要 Critic？** 它提供按状态的期望回报基准，并在片段边界自举；这样估计优势时不用只依赖完整、噪声较大的单次回报。
- **clip_range 是否硬性限制策略？** 否，它裁剪目标中的概率比，不能保证所有比值或 KL 始终在界内。
- **n_epochs 与 episode 区别？** 前者是同一批数据重复优化次数，后者是一次房间轨迹；一次 rollout 可以跨多个 episode。
- **为什么使用归一化？** 温度、功率、时钟、类别的数值尺度不同。评估复用并冻结训练统计，避免测试条件影响统计量。
- **为什么没有奖励归一化？** 本次固定使用原始奖励，便于解释及审计；这不是对其他归一化策略的否定，也没有把它纳入本次消融。
- **PPO 是否一定优于恒温器？** 不一定，需分别比较舒适、能耗和加权回报。恒温器以舒适中点为设定值，PPO 可以学习其他温度位置和动作幅度。
- **误差条是什么？** 每个模型先平均固定测试条件，再对三个独立训练种子计算样本标准差；不是把测试回合当作训练重复，也不是显著性检验。
- **为什么最后一个模型可能更差？** PPO 的更新目标与实际闭环指标存在采样/估计误差，策略可后期退化。我们预先固定末次模型，避免事后挑最好检查点。
- **为什么不使用真实建筑？** 本题允许简单热模型；模型缺少实测校准，因此不据此声称真实建筑预测精度或节能比例。
- **天气是否足够真实？** 当前是日周期函数，偏移每回合随机、每天形状重复。最终测试仍在训练分布内，没有证明跨气候和跨建筑泛化。
- **为什么环境提供围护温度？** 这是完整状态可观测的仿真假设；真实系统通常需要传感或状态估计。
- **为什么在 Mac 上做？** 两状态矩阵推进与约一万参数的网络计算较轻，本机实测可完成 21 次训练，无需 CUDA。
- **实验如何复现？** 使用 Conda ppo、锁定依赖、固定配置/条件和种子，通过命令入口重跑；保存了相关代码哈希和产物校验值。跨平台不要求逐位相同。

具体数值以 `results/summary.csv` 和最终报告为准，回答时不要把某个种子的曲线当成三种子结论。署名由三位组员补充到 `report/author.tex` 后，重新执行报告构建命令。
'''
    (ROOT / "presentation/QA_zh.md").write_text(qa)

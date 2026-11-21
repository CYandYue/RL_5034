# AIAA 5034 · PPO 房间温控项目

题目 16：**Building HVAC Control with PPO: Effect of Clip Range, GAE Lambda, and Epochs**。

在队友提供的两节点房间热模型中训练 PPO，比较 `clip_range`、`gae_lambda` 和 `n_epochs` 对舒适度、能耗和训练稳定性的影响。全部实验使用 M4 / 16 GB Mac 的 CPU，不需要 GPU 或建筑仿真服务器。

第一次了解项目请读 [项目要求与接口导读](docs/PROJECT_GUIDE_zh.md)。该导读保留最初的实施建议；实际配置以 [冻结实验协议](docs/EXPERIMENT_PROTOCOL.md) 为准，执行记录见 [TODO](TODO.md)。

详细建模解释可直接打开 [HTML 阅读版](docs/MODELING_EXPLANATION_zh.html)：公式和热网络图已内嵌，支持离线阅读及浏览器打印。对应的 [Markdown 原文](docs/MODELING_EXPLANATION_zh.md) 一并保留。

准备训练前可读 [PPO 复习与项目推进计划](docs/PPO_REVIEW_AND_PLAN_zh.html)：结合温控任务解释 GAE、裁剪目标和训练轮数，并列出默认设置、消融方案及后续阶段。

## 交付入口

- [中文结果解读（离线 HTML）](docs/RESULTS_zh.html)：先看结论，再结合曲线理解三个参数的影响。
- [英文报告 PDF](report/final_report.pdf)（10 页）及 [LaTeX 源文件](report/final_report.tex)。
- [展示 PDF](presentation/slides.pdf)（12 页）、[英文讲稿](presentation/speaker_notes.md)和[中文问答准备](presentation/QA_zh.md)。
- [结果文件说明](results/README.md)、[最终指标 CSV](results/summary.csv)及[验收记录](docs/FINAL_VERIFICATION.md)。

三位成员的姓名和学号稍后填写到 `report/author.tex`，然后重新生成报告和展示。

## 目录

```text
RL_5034/
├── materials/                # 课程 PDF；原始 ZIP 本地保留、不入库
├── RL_assignment/            # 队友热模型、76 项测试、建模文档和原始输出
├── hvac_ppo/                 # PPO 训练、共享评价、批次、汇总和文档生成
├── configs/                  # PPO 参数、7 配置矩阵、固定验证/测试条件、协议
├── tests/                    # 新增的训练/评价/协议集成测试
├── scripts/                  # screen 批次入口
├── results/                  # 实际小型结果表、原始更新诊断和 PDF/PNG 图
├── docs/                     # 中文解释、实验协议、运行及验收记录
├── report/                   # 10 页英文报告、源文件和构建清单
├── presentation/             # 12 页展示、源文件、讲稿及问答
├── runs/                     # 本地模型、归一化统计和详细日志（Git 忽略）
├── environment.yml           # Conda ppo 环境定义
└── requirements-lock.txt     # 实际 macOS arm64 依赖版本快照
```

## 安装与验证

在仓库根目录创建或激活 Conda 环境：

```bash
conda env create -f environment.yml
conda activate ppo
python -m pytest -q
python -c "from gymnasium.utils.env_checker import check_env; from hvac_env import HVAC2R2CEnv; check_env(HVAC2R2CEnv())"
python RL_assignment/scripts/dynamics_sanity.py --output-dir artifacts/env-validation --plot
```

本机 `ppo` 已安装所需依赖，无需重复创建。严格复现依赖可先 `python -m pip install -r requirements-lock.txt`，再 `python -m pip install --no-deps -e './RL_assignment[validation]' -e '.[dev,report]'`。跨平台数值可能略有差异，不要求逐位相同。

## 训练及复现

```bash
conda activate ppo
python -m hvac_ppo --help
# 仅做短跑，不读取最终测试集：
python -m hvac_ppo train --seed 0 --timesteps 102400 --output runs/pilot/default_seed_0

# 正式 7 配置 × 3 种子；每次 512,000 步。自动评价规则控制器并汇总：
screen -dmS rl5034-ppo bash scripts/run_experiments.sh 2
python -m hvac_ppo status
tail -f runs/experiment-session.log

# 实验完成后生成报告及展示（需要 TeX Live 的 pdflatex）：
python -m hvac_ppo report
```

固定场景、规则控制、独立加载、完整性校验和中断恢复见 [运行说明](docs/RUNNING_EXPERIMENTS.md)。预算由正式测试前的短跑确定，21 次训练合计 10,752,000 个交互步。完整运行经校验后跳过；未完成任务需要显式 `--restart` 归档并从种子重跑。

报告可直接从已提交的 `results/` 生成；重新绘图和审计全部训练产物需要本地 `runs/`。权重和日志未提交，已训练模型保留在本机 `runs/ablation/`。Git 提交由阶段管理，远程推送由项目负责人操作。

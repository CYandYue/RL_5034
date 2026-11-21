# AIAA 5034 · PPO 房间温控项目

题目 16：**Building HVAC Control with PPO: Effect of Clip Range, GAE Lambda, and Epochs**。

在队友提供的两节点房间热模型中训练 PPO，比较 `clip_range`、`gae_lambda` 和 `n_epochs` 对舒适度、能耗和训练稳定性的影响。全部实验使用 M4 / 16 GB Mac 的 CPU，不需要 GPU 或建筑仿真服务器。

按需要从三个入口开始：

- **交作业或改 PPT**：[交付材料导航](deliverables/README.md)，集中放报告、展示、讲稿和图素材。
- **理解项目或复现实验**：[文档导航](docs/README.md)，按学习解释、实验复现和结果解读分类。
- **查看数字或重画图**：[结果数据说明](results/README.md)，介绍原始表格、指标和绘图来源。

执行记录与组员最后需要补充的事项见 [TODO](TODO.md)。

## 交付入口

- [中文结果解读（离线 HTML）](docs/results/RESULTS_zh.html)：先看结论，再结合曲线理解三个参数的影响。
- [英文报告 PDF](deliverables/report/final_report.pdf)（10 页）及 [LaTeX 源文件](deliverables/report/final_report.tex)。
- [展示 PDF](deliverables/presentation/slides.pdf)（12 页）、[英文讲稿](deliverables/presentation/speaker_notes.md)和[中文问答准备](deliverables/presentation/QA_zh.md)。
- [绘图素材预览](deliverables/figure_assets/index.html)：11 组 PNG / 矢量 PDF，供组员修改 PPT；[图名与使用说明](deliverables/figure_assets/README.md)。
- [结果文件说明](results/README.md)、[最终指标 CSV](results/summary.csv)及[验收记录](docs/results/FINAL_VERIFICATION.md)。

三位成员的姓名和学号稍后填写到 `deliverables/report/author.tex`，然后重新生成报告和展示。

## 目录

```text
RL_5034/
├── README.md                 # 项目总入口
├── TODO.md                   # 已完成事项和最后交接
├── deliverables/             # 面向提交和展示的材料
│   ├── report/               # 10 页英文报告、LaTeX、署名和构建清单
│   ├── presentation/         # 12 页展示、讲稿和问答
│   └── figure_assets/        # PPT 用 PNG / PDF 素材及离线预览
├── docs/                     # 项目说明与阅读材料
│   ├── learning/             # 题目解析、房间建模和 PPO 复习
│   ├── experiments/          # Mac 环境、冻结协议和运行说明
│   └── results/              # 中文结果解读和验收记录
├── results/                  # 真实实验表格、诊断、原始图和校验记录
├── hvac_ppo/                 # PPO 训练、共享评价、批次、汇总和文档生成
├── RL_assignment/            # 队友交付的热模型、测试和建模依据
├── configs/                  # PPO 参数、7 配置矩阵、固定验证/测试条件、协议
├── tests/                    # 新增的训练/评价/协议集成测试
├── scripts/                  # screen 批次与图素材导出入口
├── materials/                # 课程 PDF；原始 ZIP 本地保留、不入库
├── runs/                     # 本地模型、归一化统计和详细日志（Git 忽略）
├── artifacts/                # 本地调试与独立重评输出（Git 忽略）
├── environment.yml           # Conda ppo 环境定义
├── requirements-lock.txt     # 实际 macOS arm64 依赖版本快照
└── pyproject.toml            # 项目包及测试配置
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
# 将已有图同步到 deliverables/figure_assets/：
python scripts/export_figure_assets.py
# 修改目录或文档后检查本地跳转：
python scripts/check_links.py
```

固定场景、规则控制、独立加载、完整性校验和中断恢复见 [运行说明](docs/experiments/RUNNING_EXPERIMENTS.md)。预算由正式测试前的短跑确定，21 次训练合计 10,752,000 个交互步。完整运行经校验后跳过；未完成任务需要显式 `--restart` 归档并从种子重跑。

报告可直接从已提交的 `results/` 生成；重新绘图和审计全部训练产物需要本地 `runs/`。权重和日志未提交，已训练模型保留在本机 `runs/ablation/`。Git 提交由阶段管理，远程推送由项目负责人操作。

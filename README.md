# AIAA 5034 · PPO 房间温控项目

题目 16：**Building HVAC Control with PPO: Effect of Clip Range, GAE Lambda, and Epochs**。

在简单房间热模型中训练 PPO，比较 `clip_range`、`gae_lambda` 和 `n_epochs` 对舒适度、能耗和训练稳定性的影响。计划在 M4 / 16 GB Mac 上使用 CPU 完成。

第一次了解项目请读 [项目要求、现有实现与后续工作](docs/PROJECT_GUIDE_zh.md)。执行进度见 [TODO](TODO.md)。

详细建模解释可直接打开 [HTML 阅读版](docs/MODELING_EXPLANATION_zh.html)：公式和热网络图已内嵌，支持离线阅读及浏览器打印。对应的 [Markdown 原文](docs/MODELING_EXPLANATION_zh.md) 一并保留。

准备训练前可读 [PPO 复习与项目推进计划](docs/PPO_REVIEW_AND_PLAN_zh.html)：结合温控任务解释 GAE、裁剪目标和训练轮数，并列出默认设置、消融方案及后续阶段。

## 目录

```text
RL_5034/
├── materials/                # 课程要求 PDF、原始交接 ZIP（ZIP 不入库）
├── docs/
│   └── PROJECT_GUIDE_zh.md    # 中文项目导读与待办清单
└── RL_assignment/            # 队友交付，保持原目录和文件内容
    ├── hvac_env/             # 热模型、设备、天气、指标、Gymnasium 环境
    ├── tests/                # 既有环境测试
    ├── scripts/              # 固定动作仿真脚本
    ├── docs/                 # 建模依据、参考文献和接口交接
    ├── reports/              # 队友的 Linux 验证记录
    ├── outputs/              # 已交付的温度曲线和小型 CSV
    └── pyproject.toml        # 环境包依赖
```

## 当前状态

- 已初始化本地 Git，分支为 `main`；远程仓库由项目负责人关联。
- 已创建 Conda `ppo` 环境（Python 3.12.16），队友 76 项测试和 Gymnasium 自检在 Mac 通过，见 [复验记录](docs/ENVIRONMENT_MAC.md)。
- PPO 训练、控制策略对照、超参数消融和结果汇总按 [TODO](TODO.md) 推进。
- `.gitignore` 已排除压缩包、虚拟环境、缓存和训练模型；保留课程 PDF、小型验证图和 CSV。

## 安装与验证

在仓库根目录创建或激活 Conda 环境：

```bash
conda env create -f environment.yml
conda activate ppo
python -m pytest RL_assignment/tests -q
python -c "from gymnasium.utils.env_checker import check_env; from hvac_env import HVAC2R2CEnv; check_env(HVAC2R2CEnv())"
python RL_assignment/scripts/dynamics_sanity.py --output-dir artifacts/env-validation --plot
```

本机 `ppo` 已安装所需依赖，无需重复创建。环境安装包含 Stable-Baselines3、PyTorch 和两个本地包；CPU 实验设置与锁定版本由根目录配置管理。

## 训练入口

```bash
conda activate ppo
python -m pytest -q
python -m hvac_ppo --help
python -m hvac_ppo train --seed 0 --timesteps 102400 --output runs/pilot/default_seed_0
```

固定场景、规则控制、保存/重载、正式消融与 `screen` 命令见 [运行说明](docs/RUNNING_EXPERIMENTS.md)。正式训练预算由短跑确定后再冻结。

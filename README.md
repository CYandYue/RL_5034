# AIAA 5034 · PPO 房间温控项目

题目 16：**Building HVAC Control with PPO: Effect of Clip Range, GAE Lambda, and Epochs**。

在简单房间热模型中训练 PPO，比较 `clip_range`、`gae_lambda` 和 `n_epochs` 对舒适度、能耗和训练稳定性的影响。计划在 M4 / 16 GB Mac 上使用 CPU 完成。

第一次了解项目请读 [项目要求、现有实现与后续工作](docs/PROJECT_GUIDE_zh.md)。

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
- 已导入队友环境；尚未在这台 Mac 的项目独立环境中重新运行测试。
- 尚未实现 PPO 训练、控制策略对照、超参数消融和结果汇总。
- `.gitignore` 已排除压缩包、虚拟环境、缓存和训练模型；保留课程 PDF、小型验证图和 CSV。

## 环境验证入口

以下是下一步安装后的验证方式，**本次目录整理尚未执行这些命令**。建议用 Python 3.12 的独立虚拟环境；此处的 Python 虚拟环境与 Gymnasium 仿真环境是不同概念。

```bash
cd RL_assignment
# 在已激活的 Python 3.12 虚拟环境内执行：
python -m pip install -e '.[validation]'
pytest -q
python -c "from gymnasium.utils.env_checker import check_env; from hvac_env import HVAC2R2CEnv; check_env(HVAC2R2CEnv())"
python scripts/dynamics_sanity.py --output-dir /tmp/rl5034-sanity --plot
```

后续 PPO 训练还需安装 Stable-Baselines3；它不在队友当前的运行依赖中。训练脚本、实验配置与版本记录待训练阶段加入。

# Mac 环境复验

日期：2026-11-21。设备：Apple M4 MacBook Air，16 GB 内存，macOS arm64。创建了 Conda 环境 `ppo`，解释器为 Python 3.12.16。

## 安装

```bash
conda create -n ppo python=3.12 pip -y
conda activate ppo
python -m pip install -e './RL_assignment[validation]' -e '.[dev,report]'
```

新机器也可以在仓库根目录执行 `conda env create -f environment.yml`。精确的 pip 依赖快照见 `requirements-lock.txt`，不包含本机路径的 editable 包；严格复现时先安装该文件，再以 `--no-deps -e` 安装两个本地包。

## 实际检查

| 检查 | 结果 |
|---|---|
| 队友原测试 `python -m pytest RL_assignment/tests -q` | 76 passed，0 failed，0 skipped，0.62 秒 |
| 独立 Gymnasium `check_env(HVAC2R2CEnv())` | 通过 |
| `python -m pip check` | No broken requirements found |

Gymnasium 提示直接构造的环境没有注册 `spec`，无法枚举其他渲染模式。当前模型没有声明渲染模式，温度图由独立脚本生成；这与队友 Linux 记录中的提示一致。

```bash
python -m pytest RL_assignment/tests -q
python -c "from gymnasium.utils.env_checker import check_env; from hvac_env import HVAC2R2CEnv; check_env(HVAC2R2CEnv())"
python RL_assignment/scripts/dynamics_sanity.py --output-dir artifacts/env-validation --plot
python -m pip check
```

固定动作验证输出写入本地 `artifacts/env-validation/`，不覆盖队友交付的 `RL_assignment/outputs/`。PPO 实验使用 CPU；本文件只记录环境和物理实现复验，不代表 PPO 训练结果。

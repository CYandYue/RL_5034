# AIAA 5034 · 单房间 HVAC 仿真环境

`HVAC2R2CEnv` 用两个温度状态描述室内与围护结构之间的换热。每 15 分钟输入一个供暖或制冷命令，环境返回下一时刻的温度、舒适度偏差和本步用电量。默认一条轨迹为 7 天，共 672 步。

本次完成第 16 题的房间热模型、Gymnasium 接口、舒适度与用电指标。其他组员可以在相同的天气和初态下接入 PPO，比较不同超参数的训练表现。**本仓库不训练PPO，不包含控制策略对照实验。** 题目要求与建模依据见[参考文献与证据对应](docs/REFERENCES.md)。

## 安装与验证

在本目录执行下列命令。包要求 Python 3.10 或以上，本次实际验证使用 CPython 3.12.16。`validation` 安装测试和绘图依赖，单独使用环境只需 `python -m pip install .`。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[validation]'
pytest -q
python -c "from gymnasium.utils.env_checker import check_env; from hvac_env import HVAC2R2CEnv; check_env(HVAC2R2CEnv())"
python scripts/dynamics_sanity.py --output-dir outputs --plot
```

## 一次制冷动作

```python
import numpy as np
from hvac_env import HVAC2R2CEnv

env = HVAC2R2CEnv()
obs, initial = env.reset(seed=42, options={
    "scenario": "hot",
    "weather_delta_c": 1.0,
    "indoor_init_c": 24.0,
})
action = np.array([-0.5], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)
print(info["indoor_temp_c"], info["q_hvac_kw_thermal"], info["energy_kwh_step"])
env.close()
```

此例的室温从 24.0 变为约 23.9264 °C。设备输送 −0.5 kW 热功率，消耗约 0.1667 kW 电功率，持续 0.25 h 后用电约 0.04167 kWh。

| 接口 | 含义 |
| --- | --- |
| `action` | `float32`，形状 `(1,)`。范围 `[-1, 1]`，正值供暖、零关闭、负值制冷。 |
| `obs` | `float32`，形状 `(11,)`。顺序为 `[T_i, T_e, T_o, Q_g, comfort_low, comfort_high, sin_hour, cos_hour, scenario_cold, scenario_hot, scenario_transition]`。三个温度与舒适边界用 °C，得热用热 kW，后五项无量纲。 |
| `reward` | 标量。默认值为 `-dt_h * (comfort_deviation_c + 0.1 * p_electric_kw)`，按 h、°C 和 kW 的数值计算。 |
| `terminated, truncated` | 本模型每步 `terminated=False`，时间预算用完时 `truncated=True`，随后需 `reset()`。 |
| `info` | 含采样场景、区间起止时间、步后温度、区间内实际外温与得热、请求与执行动作、热功率、电功率、单步 kWh、越界距离及奖励分项。 |

`reset(seed=123)` 可复现抽样。也可用 `options` 固定 `scenario`、`weather_delta_c`、`indoor_init_c`、`envelope_init_c`。环境接受局部配置字典或 YAML 路径，例如 `HVAC2R2CEnv({"episode_days": 1})`。全部默认参数保存在随包安装的[配置文件](hvac_env/configs/default.yaml)。

## 交接材料

| 位置 | 内容 |
| --- | --- |
| [热模型](hvac_env/thermal.py)、[设备](hvac_env/hvac.py)、[天气](hvac_env/weather.py) | 双节点热平衡、零阶保持离散化、双向热功率、合成天气和室内得热。 |
| [环境](hvac_env/env.py)、[舒适度与能耗](hvac_env/comfort.py) | Gymnasium 接口及 CVR、度时、用电量计算。精确字段、时刻及接入示例见[接口交接文档](docs/INTERFACE_HANDOFF_zh.md)。 |
| [建模说明](docs/MODEL_RATIONALE_zh.md) | 方程、单位、参数来源、可观测性与适用范围。默认热参数来自论文合成案例，天气由函数生成，电量按固定 COP 估算。 |
| [测试](tests/)、[验证报告](reports/V1_ENV_VALIDATION_zh.md) | 独立 ODE 对照、连续能量平衡、设备方向、截断和复现检查。报告列出实际命令与误差。 |
| [固定动作 CSV](outputs/dynamics_trace.csv)、[温度曲线](outputs/thermal_sanity.png) | 三种天气各 24 h 的关闭响应，以及过渡天气下各 2 h 的正负半功率响应。每行对应一个完整区间。 |

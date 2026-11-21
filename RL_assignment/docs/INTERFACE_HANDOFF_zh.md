# HVAC2R2CEnv 接口交接

## 创建与复现

在项目根目录运行 `python -m pip install .`。可直接导入 `HVAC2R2CEnv`，完整配置随包发布，工作目录可以任意选择。[默认 YAML](../hvac_env/configs/default.yaml)

```python
import numpy as np
from hvac_env import HVAC2R2CEnv

env = HVAC2R2CEnv(config={"episode_days": 1})
options = {
    "scenario": "hot",
    "weather_delta_c": 1.0,
    "indoor_init_c": 24.0,
    "envelope_init_c": 23.0,
}
obs, initial = env.reset(seed=42, options=options)
obs, reward, terminated, truncated, info = env.step(np.array([-0.5], dtype=np.float32))
```

`config` 接受 `None`、局部字典、YAML 文件路径或 `load_config()` 的结果。嵌套字段按键覆盖默认值，未知键会抛出 `ValueError`。热阻、热容、COP 和步长须大于零。容量和奖励权重允许为零。时域可以取小数天，要求 `episode_days * 24 / dt_h` 为正整数。

| `reset` 参数 | 省略时 | 显式给定时 |
| --- | --- | --- |
| `seed` | 首次由 Gymnasium 初始化发生器，随后沿用其随机流 | 非负整数重置环境的随机流 |
| `options["scenario"]` | 等概率抽取 | `"cold"`、`"hot"`、`"transition"` |
| `options["weather_delta_c"]` | 默认在 [−2, 2] °C 中均匀抽样一次 | 固定整段天气偏移，允许超出默认抽样范围 |
| `options["indoor_init_c"]` | 默认在目标中点 ±2 °C 内均匀抽样 | 固定室内节点初温，单位 °C |
| `options["envelope_init_c"]` | 按已知室内与室外初温求围护节点准平衡温度 | 固定围护节点初温，单位 °C |

固定场景和偏移即可重放天气。再固定室内初温且保持同一围护初始化规则，或显式固定两个节点，即可跨种子重放相同初态。同一配置、种子、选项与动作序列在本次测试环境中逐步精确复现。跨数值库、处理器或版本比较时用数值容差。需要随机采样动作时，Gymnasium 的 `action_space.seed(...)` 单独控制调用者的动作序列。

## 动作与观测

动作空间是 `Box(-1, 1, shape=(1,), dtype=float32)`。基础环境也接受形状相同的实数列表或数值数组。有限越界值裁剪后执行并保留请求值。标量、其他形状、字符串、NaN 和 Inf 将触发 `ValueError`，本步不会消耗步数。

观测是未归一化的 `float32` 数组，形状为 `(11,)`。代码中可导入 `OBSERVATION_FIELDS` 获取固定顺序。下表索引从 0 开始。

| 索引 | 字段 | 单位与含义 |
| ---: | --- | --- |
| 0 | `T_i` | 室内节点 °C |
| 1 | `T_e` | 围护节点 °C |
| 2 | `T_o` | 当前观测时刻的室外 °C |
| 3 | `Q_g` | 当前观测时刻的室内得热，热 kW |
| 4、5 | `comfort_low`、`comfort_high` | 目标区间闭端点，°C |
| 6、7 | `sin_hour`、`cos_hour` | $\sin(2\pi h/24)$、$\cos(2\pi h/24)$，$h$ 为日内小时 |
| 8、9、10 | `scenario_cold`、`scenario_hot`、`scenario_transition` | 按此顺序独热编码 |

初始观测对应 $t=0$。一步先用 $t_k$ 的外温、得热和动作推进，返回 $t_{k+1}$ 的室温、围护温度、天气、得热和时钟。物理状态内部保留双精度，生成观测时舍入到单精度。观测空间的温度边界取 `float32` 可表示范围，越过舒适边界的实际温度也会返回。

## 返回值和时刻

`reset` 返回 `(obs, initial)`。字典含 `scenario`、`weather_delta_c`、`step_index=0`、`time_h=0.0` 以及 `indoor_temp_c`、`envelope_temp_c`、`outdoor_temp_c`、`internal_gain_kw`、`comfort_low_c`、`comfort_high_c`。这些温度和热功率均位于初始时刻。汇总轨迹指标时从首次 `step` 开始收集。

`step` 返回 `(obs, reward, terminated, truncated, info)`。本环境无物理终止事件，`terminated` 始终为 `False`。默认第 672 步返回最后一次观测和指标，并置 `truncated=True`。首次重置前或截断后再调用 `step` 会抛出 `gymnasium.error.ResetNeeded`。下一个 episode 由调用方显式 `reset`。

| `step` 的 `info` 键 | 单位 | 所属时刻或区间 |
| --- | --- | --- |
| `scenario`、`weather_delta_c` | 字符串、°C | 本 episode 的场景与偏移 |
| `step_index` | 整数 | 已完成的区间数，首步为 1 |
| `time_h_start`、`time_h_end` | h | 本步左、右端点，从 episode 起点计时 |
| `indoor_temp_c_before` | °C | 步前室温 |
| `indoor_temp_c`、`envelope_temp_c` | °C | 步后两个节点的温度 |
| `outdoor_temp_c_applied` | °C | 左端点采样，并在本步内用于换热 |
| `internal_gain_kw_applied` | 热 kW | 左端点采样，并在本步内加到室内节点 |
| `action_requested`、`action_applied` | 无量纲 | 请求值与本区间实际执行值 |
| `q_hvac_kw_thermal` | 热 kW | 区间内净热功率，正供暖、负制冷 |
| `heating_electric_kw`、`cooling_electric_kw` | 电 kW | 区间内两种模式各自的非负耗电功率 |
| `p_electric_kw` | 电 kW | 上述两项之和 |
| `energy_kwh_step` | kWh | 本区间的电量，等于 `p_electric_kw * dt_h` |
| `comfort_low_c`、`comfort_high_c` | °C | 本场景的舒适边界 |
| `comfort_deviation_c`、`comfort_violation` | °C、布尔 | 基于步后室温的越界距离 $d$，以及 $d>10^{-8}$ °C |
| `reward_comfort`、`reward_energy`、`reward_total` | 无量纲 | 本步舒适项、电量项及其和，最后一项等于返回的 `reward` |

例如默认步长的 07:45–08:00 区间内，`internal_gain_kw_applied` 仍采用 07:45 的得热。返回观测 `obs[3]` 已含 08:00 开始的办公得热。同理，画步末室外温度应取 `obs[2]`，重算本步动力学应取 `outdoor_temp_c_applied`。

## 奖励与轨迹汇总

记 $d=\max(0,L-T_{i,k+1},T_{i,k+1}-U)$，默认 $w_c=1.0$、$w_e=0.1$。奖励为

$$
r_k=-w_c\frac{\Delta t}{1\,\mathrm{h}}\frac{d}{1\,{}^\circ\mathrm{C}}
    -w_e\frac{E_{\rm step}}{1\,\mathrm{kWh}}.
$$

舒适比例 CVR 使用 $100\operatorname{mean}(d>10^{-8})$，度时使用 $\sum_k d_k\Delta t$（°C·h），电量使用 $\sum_k P_{{\rm el},k}\Delta t$（kWh）。这里以各等长区间的步末温度代表该步的舒适状态，不能据此定位一步之内精确的越界时刻。

```python
from hvac_env.comfort import cvr_pct, degree_hours, electricity_kwh

# records contains one info dictionary per completed step of one episode.
deviations = [row["comfort_deviation_c"] for row in records]
powers = [row["p_electric_kw"] for row in records]
cvr = cvr_pct(deviations)
dh = degree_hours(deviations, env.config.dt_h)
kwh = electricity_kwh(powers, env.config.dt_h)
# The per-step energies are already integrated.
kwh_from_steps = sum(row["energy_kwh_step"] for row in records)
```

若手中只有步末室温，可用 `comfort_deviation(temperature, low, high)` 重建每步的 $d$。零步轨迹的度时和电量之和为零，CVR 需要至少一步数据。

## 配置与交接记录

| 修改项 | 对结果的影响 |
| --- | --- |
| `thermal`、HVAC 容量 | 改变换热动态或输入热功率，可能影响室温 |
| `hvac.cop_heat`、`hvac.cop_cool` | 给定相同热功率时改变用电和奖励中的能源项 |
| 天气曲线、偏移分布、`gains`、`indoor_init_spread_c` | 改变边界驱动、热负荷或起始分布 |
| 场景舒适上下限 | 改变距离、奖励和初始室温的默认抽样中点 |
| `dt_h`、`episode_days` | 改变采样/控制间隔或轨迹长度 |
| `reward.w_c`、`reward.w_e` | 只改变奖励分项和总奖励 |

正式实验可记录包版本、依赖版本、种子、固定选项、动作序列以及实际生效的配置。局部覆盖时可用 `dataclasses.asdict(env.config)` 保存合并后的完整快照，再对快照或原始 YAML 计算 SHA-256。本交付的实测对象为动力学与接口，PPO 训练、消融及策略比较由相应组员负责。[验证报告](../reports/V1_ENV_VALIDATION_zh.md)

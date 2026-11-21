# HVAC 环境 V1 验证记录

2026-11-19 在本地 Linux x86_64 上运行。全量 **76 项通过、0 项失败、0 项跳过**。独立运行 Gymnasium 自检正常返回。直接构造的环境没有注册用的 `spec`，检查器因此提示无法枚举其他渲染模式。本环境声明的渲染模式列表为空，验证图由独立脚本生成。[命令及终端输出](validation.txt)

## 复现条件与命令

| 环境 | 实测版本 |
| --- | --- |
| 解释器、包 | CPython 3.12.16，`aiaa5034-hvac-env` 1.0.0 |
| 物理计算 | NumPy 2.2.6，SciPy 1.15.3 |
| 环境与配置 | Gymnasium 1.2.3，PyYAML 6.0.3 |
| 验证 | pytest 8.4.2，Matplotlib 3.10.7 |

本机 pip 直接访问软件源时代理连接受限。本次从 PyPI 取得适用于该解释器的 wheel，逐个核对 PyPI 发布的 SHA-256，再从本地目录安装。命令中的临时目录为本次运行使用的路径。普通联网环境的安装方法见 [README](../README.md)。

```bash
/tmp/aiaa5034-hvac-venv/bin/python -m pip --disable-pip-version-check --no-cache-dir install \
    --no-index --find-links /tmp/aiaa5034-hvac-wheels-r3LzC9 -e '.[validation]'
source /tmp/aiaa5034-hvac-venv/bin/activate
pytest -q
pytest -q -s -p no:cacheprovider tests/test_thermal.py tests/test_hvac.py::test_single_step_action_direction
python -c "from gymnasium.utils.env_checker import check_env; from hvac_env import HVAC2R2CEnv; check_env(HVAC2R2CEnv())"
MPLCONFIGDIR=/tmp/aiaa5034-hvac-mpl python scripts/dynamics_sanity.py --output-dir outputs --plot
python -m pip --disable-pip-version-check --no-cache-dir check
sha256sum hvac_env/configs/default.yaml outputs/dynamics_trace.csv outputs/thermal_sanity.png
```

安装依赖检查通过。另在新虚拟环境安装了普通 wheel，从项目目录以外用 `python -I` 导入包，成功读取随包发布的 YAML、重置环境并推进一步。该环境只安装运行时依赖，独立重放生成的 CSV 用 `cmp` 检查，与交付 CSV 逐字节相同。[终端记录](validation.txt)

## 动力学数值结果

| 检查 | 本次结果 | 断言 |
| --- | --- | --- |
| 稳定性 | 连续特征值 −0.02859734、−0.17777103 h⁻¹，离散特征值 0.99287616、0.95653035 | 连续实部均负，离散模均小于 1 |
| ZOH 对独立 ODE | 最大绝对误差分别为室内 `4.96314101e-12` °C、围护 `2.22915020e-11` °C | 每个节点均低于 `1e-9` °C |
| 连续能量平衡 | 三组储热率减外部输入的残差分别为 `−1.332e-15`、`−3.997e-15`、`−1.874e-15` kW | 每组绝对残差低于 `1e-12` kW |
| 长时间被动响应 | 初态 [25, 18] °C，外温恒为 5 °C，设备与得热均为零。40 天后两节点与 5 °C 的最大距离为 `2.672e-11` °C | 最大温差逐步衰减，终值误差低于 `1e-9` °C |
| 稳态 | 室温 22 °C、室外 4 或 34 °C、得热 0.12 kW。用串联热阻反算所需设备热功率后，两节点均平衡 | 导数绝对值低于 `1e-12` °C/h，一步温度改变量低于 `1e-12` °C |

ODE 对照从 $[T_i,T_e]=[24,16]$ °C 开始，保持设备热功率 −0.6 kW、内部得热 0.27 kW、外温 5 °C 共 6 h。参考求解器 `solve_ivp(method="DOP853", rtol=1e-12, atol=1e-13)` 直接计算两个节点的热流，独立于生产代码的矩阵构造。误差取 25 个网格点（含初态）上的最大绝对值。验收阈值 `1e-9` °C 给参考积分与双精度舍入留有余量。[热模型测试](../tests/test_thermal.py)

在过渡天气、偏移 0 °C 和共同初态 $[22,18]$ °C 下，单步 0.25 h 的读数如下。此处改变一项输入以检验设备的方向与单位。[设备测试](../tests/test_hvac.py)

| 归一化动作 | 热功率 kW | 电功率 kW | 单步电量 kWh | 步末室温 °C |
| ---: | ---: | ---: | ---: | ---: |
| −0.5 | −0.5 | 0.166667 | 0.041667 | 21.866134 |
| 0 | 0 | 0 | 0 | 21.978292 |
| +0.5 | +0.5 | 0.166667 | 0.041667 | 22.090451 |

## 接口、指标与轨迹

- 三类天气均跑完默认 672 步。每次观测均属于声明的 11 维 `float32` 空间，只有末步截断。跨越每日 08:00 的测试分别检查区间起点热负荷、步末天气和新状态。还检查了动作裁剪、非法动作后的步号与所有交接字段。[API 测试](../tests/test_env_api.py)
- 同种子、配置和动作序列得到完全相同的 96 步轨迹及逐步记录。跨种子固定天气与两个初温也可精确复现。固定天气的同时仅随机化室内初温，天气观测仍逐点相同。[复现测试](../tests/test_reproducibility.py)
- 以 0.25 h 为步长，四个手算距离 [1, 0, 0, 1.5] °C 给出 CVR 50%、0.625 °C·h。电功率 [0, 1, 0.5, 0.5] kW 累计 0.5 kWh。特设稳态的两项奖励分别为 −1.0 与 −0.0125，合计 −1.0125。[指标测试](../tests/test_comfort.py)
- 实际交付 304 个区间。每种天气的 `u=0` 各占 96 步，过渡天气另有 `u=-0.5`、`+0.5` 各 8 步。初始室温 22 °C、围护 18 °C、天气偏移 0 °C，固定种子 42。CSV 的 `time_h`、`T_i_c`、`T_e_c`、`T_o_c` 是右端点值，`T_o_c` 取自单精度返回观测。`T_o_applied_c`、`Q_g_applied_kw` 和功率、电量对应刚结束的区间。PNG 尺寸为 1620 × 1080。[固定动作 CSV](../outputs/dynamics_trace.csv) · [温度曲线](../outputs/thermal_sanity.png) · [导出测试](../tests/test_demo.py)

默认配置的 SHA-256 为 `d3bebb2192820c6b2f353739eafb5202f1a72d9a4e0648011442b99cb62075e6`。交付 CSV 的 SHA-256 为 `1a492924c8c85224afbb0a27ebbf78501f606c0b66d785397a11623cfcfbec46`。

## 适用范围

验证对象是合成两节点模型的方程、数值实现、能耗代理和 Gymnasium 接口。缺少实测建筑温度、设备性能曲线和电表数据，尚不能估计真实建筑的预测误差。PPO 训练、超参数消融与策略比较属于下游成员分工，本轮状态为 **NOT RUN**。热参数和舒适区间的论文出处见[证据对应说明](../docs/REFERENCES.md)。

# 参考文献与证据对应

核对日期为 2026-11-19。下列位置可分别复核热网络、数值参数、舒适目标和软件接口。

## 第 16 题与本次交付

AIAA 5034 第 16 题为 *Building HVAC Control with PPO: Effect of Clip Range, GAE Lambda, and Epochs*。题目要求在简单房间热模型中控制供暖和制冷，训练 PPO，考察 clip range、GAE lambda 和每次更新的训练轮数，并报告舒适度越界、能耗和训练稳定性。

本次交付提供房间热模型、Gymnasium 接口及逐步舒适度与电耗数据。两节点拓扑、11 维观测、七天时域和默认奖励是本项目的实现选择。PPO 训练、消融与训练稳定性分析由后续组员完成。

## 热网络与参数

**R1**　Peder Bacher、Henrik Madsen（2011）。*Identifying suitable models for the heat dynamics of buildings*. *Energy and Buildings*, 43(7), 1511–1522。作者比较了多个候选模型，附录 A 的 TiTe 子图 (a) 给出了本项目采用的拓扑。图中室内和围护各有热容，两个热阻连接室内、围护和室外。项目所用的四个数值另见 R2，设备 COP 和本项目天气均为附加假设。[DOI](https://doi.org/10.1016/j.enbuild.2011.02.005) · [作者公开 PDF，第 11 页](https://henrikmadsen.org/wp-content/uploads/2014/05/Journal_article_-_2011_-_Identifying_suitable_models_for_the_heat_dynamics_of_buildings.pdf#page=11)

**R2**　Vahid Zamani、Shaghayegh Abtahi、Yong Li、Yuxiang Chen（2024 年卷期，2023 年在线）。*Heating and cooling supply estimation to control buildings temperature using resistor-capacitor thermal model, unscented kalman filter, and nonlinear least square method*. *Building Services Engineering Research and Technology*, 45(2), 135–160。合成案例的式 (19) 给出下表数值和单位。附录 A 式 (A1)–(A3) 给出节点和输入的位置。控制场景 Ct2 给出 1 kW 的设备容量限制。后文住宅监测案例使用另一个 10R6C 网络，见图 11 及其后说明。原文的估计方法、设备电耗及住宅拟合精度均不作为本环境的验证。 [DOI](https://doi.org/10.1177/01436244231221254) · [PMC 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10908587/)

| 原符号 | 项目配置 | 值与单位 | 连接或节点 |
| --- | --- | --- | --- |
| $R_2$ | `R_eo` | 3.02 °C/kW | 室外 ↔ 围护 |
| $R_3$ | `R_ie` | 27.81 °C/kW | 围护 ↔ 室内 |
| $C_2$ | `C_e` | 2.11 kWh/°C | 围护 |
| $C_3$ | `C_i` | 1.11 kWh/°C | 室内 |

## 舒适区间

**R3**　Antonio Manjavacas、Alejandro Campoy-Nieves、Javier Jiménez-Raboso、Miguel Molina-Solana、Juan Gómez-Romero（2024）。*An experimental evaluation of deep reinforcement learning algorithms for HVAC control*. *Artificial Intelligence Review*, 57, 173。第 5.2 节的舒适度评价采用五区办公室在 6 至 9 月的 [23, 26] °C，其余月份为 [20, 23.5] °C，并统计越界时间比例和偏离幅度。这里使用的是该节评价区间。将月份映射到三类合成天气、按步末温度积分度时、选用奖励系数均为本项目决定。该项实验设置的证据范围限于文中的办公室案例。[DOI](https://doi.org/10.1007/s10462-024-10819-x) · [期刊全文，第 5.2 节](https://link.springer.com/article/10.1007/s10462-024-10819-x#Sec15)

## 软件接口

**R4**　Farama Foundation。*Make your own custom environment*、*Env*、*Utility functions*, Gymnasium 在线文档（2026 年访问）。用于核对 `super().reset(seed=seed)`、`self.np_random`、观测与动作空间、五元组返回以及 `check_env` 检查。本项目用 Gymnasium 1.2.3 执行了自检。观测列、动作物理意义和本项目的奖励定义来自实现。[自定义环境教程](https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/) · [Env API](https://gymnasium.farama.org/api/env/) · [检查工具](https://gymnasium.farama.org/api/utils/)

**R5**　SciPy 开发者。*scipy.signal.cont2discrete*, SciPy 1.15.3 Reference Guide（2026 年访问）。用于核对连续状态空间输入 `(A, B, C, D)`、步长参数及 `method="zoh"`。输入在每步内保持恒定是本项目的采样约定，温度误差由项目另行实测。[对应版本的 API 文档](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.signal.cont2discrete.html)

## 本项目产生的证据

合成天气的均值、振幅和偏移分布，得热日程、初态分布、对称容量、固定 COP、步长、时域及奖励权重集中在[默认配置](../hvac_env/configs/default.yaml)。两节点温度误差、守恒残差与 API 检查可由[验证报告](../reports/V1_ENV_VALIDATION_zh.md)中的命令复现。真实建筑温度预测、实际电表读数和学习策略效果尚无本项目数据验证。

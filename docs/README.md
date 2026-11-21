# 项目文档导航

从 [项目总览](../README.md) 返回仓库入口。报告、展示和图素材统一放在 [deliverables/](../deliverables/README.md)；本目录放解释、操作说明和结果解读。

## learning/：理解题目、模型和 PPO

| 文档 | 建议用途 |
| --- | --- |
| [项目要求与接口导读](learning/PROJECT_GUIDE_zh.md) | 第一次了解题目；包含课程要求和最初实施建议。实际配置以冻结协议为准。 |
| [房间建模解释 · HTML](learning/MODELING_EXPLANATION_zh.html) | 逐项理解 11 维观测、热阻热容、动作和奖励；公式与热网络图可离线阅读。 |
| [房间建模解释 · Markdown](learning/MODELING_EXPLANATION_zh.md) | 保留解释原文，方便检索和版本比较。 |
| [PPO 复习与项目计划 · HTML](learning/PPO_REVIEW_AND_PLAN_zh.html) | 结合温控复习 PPO、GAE、clip range 和 epochs。 |

## experiments/：环境与复现

| 文档 | 建议用途 |
| --- | --- |
| [Mac 环境记录](experiments/ENVIRONMENT_MAC.md) | Conda ppo、依赖版本及环境复验。 |
| [冻结实验协议](experiments/EXPERIMENT_PROTOCOL.md) | 正式训练前确定的预算、参数、种子、评价和模型选择规则。 |
| [训练与复现入口](experiments/RUNNING_EXPERIMENTS.md) | 训练、screen、独立加载、汇总和生成报告的命令。 |

所有命令均在仓库根目录执行，文件被分组后不需要切换命令的工作目录。

修改目录或跳转链接后，可运行 `python scripts/check_links.py` 检查项目阅读材料和交付物中的本地 Markdown / HTML 链接。

## results/：阅读结果与验收

| 文档 | 建议用途 |
| --- | --- |
| [中文结果解读 · HTML](results/RESULTS_zh.html) | 快速了解实际发现，再对照内嵌图表阅读；可离线打开。 |
| [最终验收记录](results/FINAL_VERIFICATION.md) | 测试、模型重载、预算、哈希、文档页数及目录整理后的复核。 |

逐回合数据、跨种子汇总和原始图位于仓库根目录的 [results/](../results/README.md)，与本目录的结果说明分开。队友原始的 [建模依据](../RL_assignment/docs/MODEL_RATIONALE_zh.md)、[接口交接](../RL_assignment/docs/INTERFACE_HANDOFF_zh.md)和[参考文献](../RL_assignment/docs/REFERENCES.md)仍随环境包保留。

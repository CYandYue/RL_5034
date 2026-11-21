# 训练与复现入口

所有命令在仓库根目录、`conda activate ppo` 后执行。`python -m hvac_ppo --help` 查看入口。

## 已验证

原有环境测试加上新增训练/评估测试共 86 项通过。新增检查包含固定条件重放、独立单位算例、保存/重载、评估冻结统计、同种子 CPU 训练复现及运行完整性检查。

## 固定场景与控制器

```bash
python -m hvac_ppo prepare
python -m hvac_ppo baseline --cases configs/scenarios/validation.json --output artifacts/baselines-validation
```

验证集每类天气 5 个条件（共 15 个），测试集每类 10 个（共 30 个）。两组物理条件无重复。文件保存实际天气偏移和两个初温；生成种子不等于训练种子。

规则对照为：关闭空调；以及以舒适中点为设定值、上下各 0.5°C 死区的满功率继电控制。后者低于死区供暖，高于死区制冷，死区内关闭。该设置在正式测试前确定，不以测试结果调参。

## 单次短跑

```bash
python -m hvac_ppo train --seed 0 --timesteps 102400 --output runs/pilot/default_seed_0
```

短跑不传入 `--test`，只用验证集。网络为独立策略/价值两层 64 单元 MLP，CPU、Tanh、高斯策略；观测归一化，奖励保持原始单位。验证在完整 PPO 更新后执行，评估不更新网络及归一化统计。

## 正式批次与 screen

正式预算经短跑确认并写入 `configs/ppo/base.yaml` 后：

```bash
screen -dmS rl5034-ppo bash scripts/run_experiments.sh 2
screen -ls
tail -f runs/experiment-session.log
python -m hvac_ppo status
```

`2` 表示两个独立 CPU 训练进程，每个 PyTorch 进程使用一个计算线程。`screen -r rl5034-ppo` 可接入，按 `Ctrl-A` 然后 `D` 脱离。批次结束会话自动退出；最终状态以磁盘上的 `runs/ablation/batch_manifest.json` 为准。

每个运行存于 `runs/ablation/<configuration>/seed_<seed>/`，控制台日志见 `runs/ablation/_console/`。批次结束后自动评估规则控制器并汇总结果。

运行目录含模型、归一化统计、配置、依赖/代码版本、定期验证、最终测试及产物 SHA-256。`progress.csv` 含每轮更新后的 KL、裁剪比例、价值误差、熵损失、梯度更新次数和采样动作饱和比例等。七天时间结束保持 Gymnasium 的截断语义，使用 SB3 自举处理。

恢复批次时，只有配置、场景、相关训练源码和产物哈希均一致的完整运行会被跳过。未完成或不兼容运行默认报错；加 `--restart` 会先归档原目录，再从种子重新训练。它不声称精确恢复中途的采样器和随机数状态。每次周期验证也保存最新模型及归一化统计，方便诊断。

```bash
python -m hvac_ppo ablate --jobs 2
# 需要归档并重跑未完成任务时：
python -m hvac_ppo ablate --jobs 2 --restart
```

## 加载并独立评估

```bash
python -m hvac_ppo evaluate --run runs/ablation/default/seed_0 \
  --cases configs/scenarios/test.json --output artifacts/reloaded-default
python -m hvac_ppo summarize
```

最终比较使用固定采样预算的末次模型。测试条件不参与训练、检查点选择或参数调节。训练随机性的均值/标准差先按每个模型的固定场景均值计算，再跨训练种子汇总，不能把测试回合当作独立训练。

大模型与日志由 Git 忽略，小型结果表和报告图放在 `results/`，适合入库。模型可在本机保留，另一台机器按照配置和种子重新训练。

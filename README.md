# 猴子反转学习：RW 与 GRU

用自写代码拟合同一只猴子的选择行为，比较 RW 与两个隐藏单元的 GRU，并生成教材图片。

```bash
uv sync --locked
uv run rl-rnn-example
```

配置直接修改 `src/rl_rnn_example/main.py`。默认使用猴子 V，每个完整 block 保留原始第 11–70 次；按整 block 固定划分训练、验证和测试集。

默认结果目录为 `results/monkey_V_seed0/`：

- `experiment.json`：配置、各组 block 标识、RW 参数、测试指标和 GRU 训练记录。
- `gru_weights.pt`：根据验证损失选出的 GRU 权重。
- `predictions.json`：所有测试 block 的真实行为、反转位置、两模型逐试次概率及状态。
- `figures/`：行为、预测对比、状态轨迹、RW/GRU 更新箭头和训练曲线；每张同时保存 PNG 与 SVG。

将 `FIT_MODELS` 改成 `False` 可读取该目录的结果并重新画图。`PLOT_BLOCK_INDEX` 选择展示哪个测试 block，`GRID_SIZE` 控制箭头图的网格密度。再次拟合会更新当前输出目录；保留另一次实验时修改 `OUTPUT_DIR`。

| 文件 | 用途 |
| --- | --- |
| `dataset.py` | 读取行为，整理 block，并保留原始反转试次编号 |
| `rw.py`、`gru.py` | 预测选择、更新内部状态 |
| `training.py`、`metrics.py` | 划分、拟合、评价并记录训练损失 |
| `analysis.py` | 整理预测、计算更新箭头、保存和加载结果 |
| `plotting.py` | 生成图片 |
| `main.py` | 配置与完整流程 |

模型先预测当前选择，再读取当前动作和奖励。每个保留 block 独立从零状态开始；反转点不重置。箭头表示指定输入下的一次状态变化，背景色表示选择动作 0 的概率；灰色轨迹来自该测试 block 的真实选择和奖励。网格中的其他位置是对模型的探查，不能视作真实访问过的状态。GRU 坐标是隐藏状态，不直接等同于 Q 值；两个模型的箭头长度不宜直接比较。

当前采用一次固定划分，结果用于教学示例，不代表完整论文的交叉验证复现。图中的动作 0/1 在 what block 表示图像选项，在 where block 表示位置选项。

检查时序、单步更新和保存重载：`uv run python -m unittest discover -s tests -v`。

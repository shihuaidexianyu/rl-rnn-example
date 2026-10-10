# 猴子反转学习：RW 与 GRU

用自写代码比较两单元 GRU 与经典 RW 模型的测试预测表现，并绘制整个测试集上的条件动力学。

```bash
uv sync --locked
uv run rl-rnn-example
```

配置直接修改 `src/rl_rnn_example/main.py`。当前 `FIT_MODELS=False`，读取已有权重重画；首次运行没有结果文件时，先改为 `True` 拟合一次。默认使用猴子 V，每个完整 block 保留原始第 11–70 次，按整 block 固定划分训练、验证和测试集。

默认结果目录为 `results/monkey_V_seed0/`：

- `experiment.json`：配置、各组 block 标识、RW 参数、测试指标和 GRU 训练记录。
- `gru_weights.pt`：根据验证损失选出的 GRU 权重。
- `predictions.json`：所有测试 block 的真实行为、反转位置、两模型逐试次概率及状态。
- `figures/model_comparison.png`：主图一，同一测试集上的平均负对数似然（越低越好）和选择预测准确率（越高越好），按试次数加权。
- `figures/dynamics_comparison.png`：主图二，整个测试集的动力学对照；上排 RW、下排 GRU，四列分别固定一种动作与奖励组合。
- 两张主图采用简洁面板排版；数据来源、符号解释和方法说明另存于对应的 `*.caption.txt` 图注文件。
- 每张图同时保存 PNG 与 SVG。

动力学由固定模型参数决定，测试集提供实际访问的状态与反馈事件。默认 `MAX_ARROWS_PER_CONDITION=None`，显示全部测试事件；设为正整数时才按条件抽样，`DYNAMICS_SAMPLE_SEED` 保证可重复。`GRID_SIZE` 控制背景网格密度。

`PLOT_SUPPLEMENTARY=False` 时只生成两张主图。设为 `True` 可补充生成单模型动力学图、训练曲线、单 block 示例和单次更新图。`PLOT_BLOCK_INDEX`、`ONE_STEP_TRIAL` 以及 `PROBE_*` 配置仅用于这些补充示例，不影响主图。

再次拟合会更新当前输出目录；保留另一次实验时修改 `OUTPUT_DIR`。

| 文件 | 用途 |
| --- | --- |
| `dataset.py` | 读取行为，整理 block，并保留原始反转试次编号 |
| `rw.py`、`gru.py` | 预测选择、更新内部状态 |
| `training.py`、`metrics.py` | 划分、拟合、评价并记录训练损失 |
| `analysis.py` | 汇总测试指标、计算一步更新和多 block 动力学、保存和加载结果 |
| `plotting.py` | 生成图片 |
| `main.py` | 配置与完整流程 |

模型先预测当前选择，再读取当前动作和奖励。每个保留 block 独立从零状态开始；反转点不重置。

动力学图的背景范围由所有测试 block 的状态共同确定，包括每个 block 的末状态。状态差分先在各 block 内计算，再按动作和奖励汇总，不连接不同 block 的首尾。箭头抽样不改变背景范围，RW 与 GRU 使用相同的一批事件。多个 block 混合 what/where 时，图中统一写“选项 0/1”，并说明两种任务的编码。

背景色和细等值线表示该条件下的一次状态更新幅度；黑箭头表示实际行为驱动的模型更新。橙色虚线表示两个动作等概率。同一模型四幅图共用幅度色标；不同模型的状态单位不同，不能据颜色或箭头长度比较学习率。网格是对假设状态的探查，不代表实际访问过的位置；等值线不是轨迹。

比较只针对当前被试、一次固定划分和 RW 这个基线；结果若支持，可以写“本例中 GRU 的测试预测优于 RW”，不能直接推广成 tinyRNN 优于所有经典模型。预测准确率也不等于获得奖励的比例。目前不做额外超参数搜索或嵌套交叉验证。

GRU 使用普通线性读出，原图使用对角读出；这里不绘制白色吸引子标记，也未做原文图 4b 的动力学回归。GRU 的 h1、h2 不直接等于两个 Q 值，其等概率线由实际读出权重决定，不能预设为 h1=h2。

图中的动作 0/1 对应原文的 A1/A2；在 what block 表示图像选项，在 where block 表示位置选项。

检查时序、单步更新和保存重载：`uv run python -m unittest discover -s tests -v`。

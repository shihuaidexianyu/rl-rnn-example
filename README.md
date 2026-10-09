# 猴子反转学习：RW 与 GRU

这是供自己填写实现的代码框架。已写好函数接口、输入输出约定和中文步骤；数据处理和核心算法中的 `TODO` 与 `NotImplementedError` 等待实现。

建议按这个顺序填写：

1. `dataset.py`：先完成 `load_session` 整理一个文件，再完成 `load_blocks` 汇总一只猴子的数据。
2. `rw.py`：计算选择概率、更新 Q 值、处理一个 block。
3. `metrics.py`：计算负对数似然和准确率。
4. `gru.py`：建立网络，处理输入与预测的时间对齐。
5. `training.py`：按完整 block 划分数据，分别拟合两个模型。
6. `main.py`：查看完整调用顺序，在文件顶部修改实验配置。

以上文件均位于 `src/rl_rnn_example/`。本框架一次拟合一只猴子，每个完整 block 保留原始第 11–70 次；GRU 使用 `[action, reward]` 两维输入。先实现一次固定划分，暂不加入嵌套交叉验证、神经数据分析和绘图。

```bash
uv sync --locked
uv run rl-rnn-example
```

当前运行会在首个未实现的函数处给出中文提示。完成所有待填部分后，该入口会使用同一批测试 block 比较两个模型。

"""第五步：划分完整 block，并分别拟合 RW 和 GRU。

多个训练 block 共同拟合一组参数；每个 block 的内部状态独立初始化。
先完成一次固定的训练/验证/测试划分，暂不加入原论文的嵌套交叉验证。
"""


def split_blocks(
    blocks: list[dict],
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[dict], list[dict], list[dict]]:
    """按完整 block 划分数据，返回训练集、验证集、测试集。

    默认各取约 10% 的 block 作为验证和测试，剩余约 80% 用于训练。
    相同输入和随机种子应得到相同划分。
    """
    # TODO 1：检查所有 block 来自同一只猴子，且 session/block 编号没有重复。
    # TODO 2：检查两个比例均大于 0、总和小于 1，并确保三个子集都非空。
    # TODO 3：使用局部随机数生成器打乱 block 索引，不打乱块内试次。
    # TODO 4：计算各子集的 block 数量，按索引提取并返回三个列表。
    # 同一个 block 只能出现在一个子集中；不要修改原 blocks 列表。
    raise NotImplementedError("请实现 split_blocks：按完整 block 划分训练、验证和测试数据。")


def fit_rw(model, train_blocks: list[dict]) -> None:
    """拟合 RW 的 alpha 和 iTemp，将结果写回 model。

    目标是训练集所有试次的平均负对数似然。
    这里只使用训练集，验证和测试数据不参与参数搜索。
    """
    # TODO 1：定义目标函数，接收候选 alpha、iTemp。
    # TODO 2：设置候选参数，逐个训练 block 重新预测，汇总平均负对数似然。
    # 每次评估新参数时，各 block 的 Q 值都需要从零开始。
    # TODO 3：使用 scipy.optimize.minimize 搜索较小的损失。
    # 将 alpha 限制在 0–1 之间，iTemp 限制为正数，并检查优化是否成功。
    # TODO 4：把最终拟合出的参数写入 model.alpha 和 model.iTemp。
    # 待拟合的是这两个参数，不是每次试次的 Q 值。
    raise NotImplementedError("请实现 fit_rw：通过选择预测损失拟合两个参数。")


def fit_gru(
    model,
    train_blocks: list[dict],
    validation_blocks: list[dict],
    learning_rate: float = 0.005,
    max_epochs: int = 1000,
    patience: int = 100,
) -> None:
    """拟合 GRU 参数，并将验证损失最小的参数恢复到 model。

    learning_rate 控制优化器如何调整参数，与 RW 中的 alpha 含义不同。
    一轮训练使用全部训练 block，汇总它们的预测损失后更新一次参数。
    """
    # TODO 1：把训练 block 整理为 inputs (T, B, 2) 和 targets (T, B)。
    # inputs 的 dtype 和 device 与模型参数一致，targets 使用 torch.long 且在同一设备。
    # targets 存放整数动作编号；保留块内时间顺序，验证数据也按相同方式整理。
    # 当前所有 block 等长且有效，第一版不需要加入填充和 mask 接口。
    # TODO 2：创建优化器，例如 AdamW，并明确设置 weight_decay=0。
    # 本框架先只使用预测损失，不另加 L1 惩罚或权重衰减。
    # TODO 3：每轮进入训练模式，清除旧梯度，调用 model(inputs) 得到 logits。
    # 将 logits 整理为 (T*B, 2)，targets 整理为 (T*B,)，计算平均交叉熵。
    # 不要使用返回 NumPy 的 predict_block 来训练，也不要先对 logits 做 softmax。
    # TODO 4：执行反向传播和一次参数更新。
    # TODO 5：切换到评估模式并关闭梯度，在验证 block 上计算平均损失。
    # 验证表现变好时，深拷贝当前参数；连续 patience 轮未改善时提前停止。
    # TODO 6：结束时加载保存的最佳参数，而不是直接保留最后一轮参数。
    # 测试集只在 main.py 的最终评估时使用。
    raise NotImplementedError("请实现 fit_gru：训练网络并根据验证损失保存最佳参数。")

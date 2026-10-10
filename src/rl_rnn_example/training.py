"""划分完整 block，并分别拟合 RW 和 GRU。

多个训练 block 共同拟合一组参数；每个 block 的内部状态独立初始化。
当前采用一次固定的训练/验证/测试划分，不做嵌套交叉验证。
"""

import copy

import numpy as np
import torch
from scipy.optimize import minimize

from .metrics import negative_log_likelihood


def split_blocks(
    blocks: list[dict],
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[dict], list[dict], list[dict]]:
    """按完整 block 划分数据，返回训练集、验证集、测试集。

    默认各取约 10% 的 block 作为验证和测试，剩余约 80% 用于训练。
    相同输入和随机种子应得到相同划分。
    比例不合法或按当前比例划分后有空集合时，会给出明确错误。
    """
    if not (
        0 < validation_fraction < 1
        and 0 < test_fraction < 1
        and validation_fraction + test_fraction < 1
    ):
        raise ValueError("验证集和测试集比例都必须在 0–1 之间，且两者之和小于 1。")

    # 使用局部随机数生成器打乱 block 索引，不打乱块内试次。
    rng = np.random.default_rng(seed)
    # 计算各子集的 block 数量，按索引提取并返回三个列表。
    # 同一个 block 只能出现在一个子集中；不要修改原 blocks 列表。
    n_blocks = len(blocks)
    n_validation = int(n_blocks * validation_fraction)
    n_test = int(n_blocks * test_fraction)
    n_train = n_blocks - n_validation - n_test
    if min(n_train, n_validation, n_test) == 0:
        raise ValueError(
            f"当前 {n_blocks} 个 block 按所设比例划分后，"
            f"训练/验证/测试数量为 {n_train}/{n_validation}/{n_test}，存在空集合。"
            "请增加完整 block 数量，或调整验证集和测试集比例。"
        )

    indices = np.arange(n_blocks)
    rng.shuffle(indices)

    train_indices = indices[:n_train]
    validation_indices = indices[n_train : n_train + n_validation]
    test_indices = indices[n_train + n_validation :]

    train_blocks_list = [blocks[i] for i in train_indices]
    validation_blocks_list = [blocks[i] for i in validation_indices]
    test_blocks_list = [blocks[i] for i in test_indices]

    return train_blocks_list, validation_blocks_list, test_blocks_list


def fit_rw(model, train_blocks: list[dict]) -> None:
    """拟合 RW 的 alpha 和 iTemp，将结果写回 model。

    目标是训练集所有试次的平均负对数似然。
    这里只使用训练集，验证和测试数据不参与参数搜索。
    """

    # 定义目标函数，接收候选 alpha、iTemp。
    # 设置候选参数，逐个训练 block 重新预测，汇总平均负对数似然。
    # 每次评估新参数时，各 block 的 Q 值都需要从零开始。
    def objective(params):
        alpha, iTemp = params
        model.alpha = alpha
        model.iTemp = iTemp
        total_nll = 0.0
        total_trials = 0
        for block in train_blocks:
            probabilities = model.predict_block(block["actions"], block["rewards"])[
                "probabilities"
            ]
            n_trials = len(block["actions"])
            total_nll += (
                negative_log_likelihood(probabilities, block["actions"]) * n_trials
            )
            total_trials += n_trials
        return total_nll / total_trials

    # 使用 scipy.optimize.minimize 搜索较小的损失。
    # 将 alpha 限制在 0–1 之间，iTemp 限制为正数，并检查优化是否成功。
    result = minimize(
        objective,
        x0=[0.5, 1.0],  # 初始猜测值
        bounds=[(0, 1), (1e-8, None)],
        method="L-BFGS-B",
    )
    if not result.success:
        raise ValueError("优化失败，无法拟合 RW 参数。")
    # 把最终拟合出的参数写入 model.alpha 和 model.iTemp。
    # 待拟合的是这两个参数，不是每次试次的 Q 值。
    alpha, iTemp = result.x
    model.alpha = alpha
    model.iTemp = iTemp


def fit_gru(
    model,
    train_blocks: list[dict],
    validation_blocks: list[dict],
    learning_rate: float = 0.005,
    max_epochs: int = 1000,
    patience: int = 100,
) -> dict:
    """拟合 GRU 参数，并将验证损失最小的参数恢复到 model。

    learning_rate 控制优化器如何调整参数，与 RW 中的 alpha 含义不同。
    一轮训练使用全部训练 block，汇总它们的预测损失后更新一次参数。

    max_epochs 和 patience 必须为正整数，训练集与验证集都不能为空。
    返回训练记录字典：
        epoch: 从 1 开始的实际训练轮次列表。
        train_nll: 每轮参数更新前的训练集平均负对数似然。
        validation_nll: 每轮参数更新后的验证集平均负对数似然。
        best_epoch: 验证损失最小时对应的轮次，从 1 开始。
    两条损失曲线的计算时点不同；返回后 model 使用 best_epoch 的参数。
    """
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs <= 0:
        raise ValueError("max_epochs 必须是正整数。")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
        raise ValueError("patience 必须是正整数。")
    if not train_blocks or not validation_blocks:
        raise ValueError("训练集和验证集都不能为空，请检查完整 block 的划分结果。")

    # 把训练 block 整理为 inputs (T, B, 2) 和 targets (T, B)。
    # inputs 的 dtype 和 device 与模型参数一致，targets 使用 torch.long 且在同一设备。
    # targets 存放整数动作编号；保留块内时间顺序，验证数据也按相同方式整理。
    # 当前所有 block 等长且有效，第一版不需要加入填充和 mask 接口。
    param = next(model.parameters())
    device, dtype = param.device, param.dtype

    def prepare(blocks):
        inputs = np.stack(
            [
                np.stack([block["actions"], block["rewards"]], axis=-1)
                for block in blocks
            ],
            axis=1,
        )  # (T, B, 2)，与模型约定的时间在前一致
        targets = np.stack([block["actions"] for block in blocks], axis=1)  # (T, B)
        return (
            torch.tensor(inputs, dtype=dtype, device=device),
            torch.tensor(targets, dtype=torch.long, device=device),
        )

    train_inputs, train_targets = prepare(train_blocks)
    validation_inputs, validation_targets = prepare(validation_blocks)
    # 创建优化器，例如 AdamW，并明确设置 weight_decay=0。
    # 本框架先只使用预测损失，不另加 L1 惩罚或权重衰减。
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss()
    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {
        "epoch": [],
        "train_nll": [],
        "validation_nll": [],
        "best_epoch": 0,
    }
    # 每轮进入训练模式，清除旧梯度，调用 model(inputs) 得到 logits。
    # 将 logits 整理为 (T*B, 2)，targets 整理为 (T*B,)，计算平均交叉熵。
    # 不要使用返回 NumPy 的 predict_block 来训练，也不要先对 logits 做 softmax。
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_inputs)["logits"]  # (T, B, 2)
        loss = criterion(logits.reshape(-1, 2), train_targets.reshape(-1))
        # 执行反向传播和一次参数更新。
        loss.backward()
        optimizer.step()
        # 切换到评估模式并关闭梯度，在验证 block 上计算平均损失。
        # 验证表现变好时，深拷贝当前参数；连续 patience 轮未改善时提前停止。
        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_inputs)["logits"]
            validation_loss = criterion(
                validation_logits.reshape(-1, 2), validation_targets.reshape(-1)
            ).item()
        # 保存标量，不保留计算图；提前停止的最后一轮也记录在内。
        history["epoch"].append(epoch + 1)
        history["train_nll"].append(loss.item())
        history["validation_nll"].append(validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    # 结束时加载保存的最佳参数，而不是直接保留最后一轮参数。
    # 测试集只在 main.py 的最终评估时使用。
    if best_state is not None:
        model.load_state_dict(best_state)
    return history

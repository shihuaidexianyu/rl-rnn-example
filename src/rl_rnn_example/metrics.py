"""第三步：用同一种标准评价 RW 和 GRU 的预测。

负对数似然越小越好；准确率表示最大概率对应的动作与真实选择相同的比例。
真实选择是预测目标，是否获得奖励不直接决定这次预测是否正确。
"""

import numpy as np


def negative_log_likelihood(probabilities: np.ndarray, actions: np.ndarray) -> float:
    """返回一个 block 的平均负对数似然。

    probabilities 的形状为 (T, 2)，actions 的形状为 (T,)。
    每次损失为 -ln(P(真实选择))，最后对 T 次试次取平均。
    """
    # TODO 1：取出每行中真实动作对应的概率。
    # TODO 2：给概率设置很小的正数下界，避免计算 log(0)。
    # TODO 3：计算负的自然对数，并返回平均值。
    raise NotImplementedError("请实现 negative_log_likelihood：计算平均预测损失。")


def accuracy(probabilities: np.ndarray, actions: np.ndarray) -> float:
    """返回预测准确率，范围为 0–1。"""
    # TODO 1：每行取概率较大的动作编号；平局时统一选择编号 0。
    # TODO 2：与真实 actions 比较，计算相同的比例。
    raise NotImplementedError("请实现 accuracy：计算选择预测正确的比例。")


def evaluate(model, blocks: list[dict]) -> dict:
    """汇总模型在多个 block 上的表现，不修改模型参数。

    model 可以是 RWModel 或 GRUModel，均提供 predict_block 方法。
    返回字典包含 n_trials（总试次数）、nll（平均损失）、accuracy（准确率）。
    """
    # TODO 1：逐个 block 调用 model.predict_block(actions, rewards)。
    # TODO 2：使用上面的两个函数计算每个 block 的表现。
    # TODO 3：按试次数加权汇总，返回所有试次上的平均指标。
    # 不要把 block 首尾拼接后预测，否则会让记忆跨越 block 边界。
    raise NotImplementedError("请实现 evaluate：汇总多个独立 block 的预测表现。")

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
    # 取出每行中真实动作对应的概率。
    true_probs = probabilities[np.arange(len(actions)), actions]
    # 给概率设置很小的正数下界，避免计算 log(0)。
    true_probs = np.clip(true_probs, a_min=1e-10, a_max=None)
    # 计算负的自然对数，并返回平均值。
    return -np.mean(np.log(true_probs))


def accuracy(probabilities: np.ndarray, actions: np.ndarray) -> float:
    """返回预测准确率，范围为 0–1。"""
    # 每行取概率较大的动作编号；平局时统一选择编号 0。
    predicted_actions = np.argmax(probabilities, axis=1)
    # 与真实 actions 比较，计算相同的比例。
    return np.mean(predicted_actions == actions)


def evaluate(model, blocks: list[dict]) -> dict:
    """汇总模型在多个 block 上的表现，不修改模型参数。

    model 可以是 RWModel 或 GRUModel，均提供 predict_block 方法。
    返回字典包含 n_trials（总试次数）、nll（平均损失）、accuracy（准确率）。
    """
    # 逐个 block 调用 model.predict_block(actions, rewards)。
    all_nll = []
    all_accuracy = []
    total_trials = 0
    for block in blocks:
        actions = block["actions"]
        rewards = block["rewards"]
        result = model.predict_block(actions, rewards)
        probabilities = result["probabilities"]
        # 使用上面的两个函数计算每个 block 的表现。
        nll = negative_log_likelihood(probabilities, actions)
        acc = accuracy(probabilities, actions)
        # 按试次数加权汇总，返回所有试次上的平均指标。
        # 不要把 block 首尾拼接后预测，否则会让记忆跨越 block 边界。
        all_nll.append(nll * len(actions))
        all_accuracy.append(acc * len(actions))
        total_trials += len(actions)

    return {
        "n_trials": total_trials,
        "nll": float(np.sum(all_nll) / total_trials),
        "accuracy": float(np.sum(all_accuracy) / total_trials),
    }

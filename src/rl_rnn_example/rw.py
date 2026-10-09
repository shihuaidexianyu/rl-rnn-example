"""第二步：手写反转学习中的 RW 模型。

固定参数 alpha、iTemp 时，模型根据每次真实选择和奖励更新 Q 值。
参数拟合放在 training.py；本文件只负责价值更新和选择概率计算。
"""

import numpy as np


class RWModel:
    """用两个价值 Q(A)、Q(B) 描述选择行为。

    alpha 是行为模型的学习率，控制奖励对价值估计的影响。
    iTemp 是逆温度，控制模型对价值差异的敏感程度。
    这两个参数在一个 block 的前向计算过程中保持不变。
    """

    def __init__(self, alpha: float = 0.5, iTemp: float = 5.0):
        self.alpha = alpha
        self.iTemp = iTemp

    def choice_probabilities(self, values: np.ndarray) -> np.ndarray:
        """输入形状为 (2,) 的 Q 值，返回形状为 (2,) 的选择概率。"""
        # 计算 softmax(iTemp * values)。
        # 先减去最大分数再取指数，可以避免数值溢出。
        # 返回顺序固定为 [P(action=0), P(action=1)]，两项之和为 1。
        values = np.array(values, dtype=float)
        max_value = np.max(values)
        exp_values = np.exp(self.iTemp * (values - max_value))
        probabilities = exp_values / np.sum(exp_values)
        return probabilities

    def update_values(self, values: np.ndarray, action: int, reward: int) -> np.ndarray:
        """返回更新后的 Q 值，形状为 (2,)，保持传入的 values 不变。"""
        # 复制旧 values，以免修改已保存的历史状态。
        new_values = values.copy()
        # 只更新本次选择的选项：Q新 = Q旧 + alpha * (reward - Q旧)。
        new_values[action] += self.alpha * (reward - new_values[action])
        # 未选择的选项保持不变；不要根据反转标记直接重置 Q 值。
        return new_values

    def predict_block(self, actions: np.ndarray, rewards: np.ndarray) -> dict:
        """处理一个 block，返回预测概率和价值轨迹。

        输入：actions、rewards 均为长度 T 的一维数组。
        返回：
            probabilities: NumPy 数组，形状为 (T, 2)。
            states: NumPy 数组，形状为 (T, 2)，每行是 [Q(A), Q(B)]。

        时序约定：第 t 行状态和概率来自读取第 t 次事件之前。
        即先预测 actions[t]，再读取 actions[t]、rewards[t] 更新价值。
        返回 T 行，不包含读完最后一次事件后的额外状态。
        """
        # 每次调用都从 Q=[0, 0] 开始，不继承上一个 block 的状态。
        values = np.array([0.0, 0.0], dtype=float)
        probabilities = []
        states = []
        for t in range(len(actions)):
            # 逐次保存当前 Q 和 choice_probabilities(Q)。
            states.append(values.copy())
            probabilities.append(self.choice_probabilities(values))
            # 使用真实动作和奖励调用 update_values，进入下一次试次。
            values = self.update_values(values, actions[t], rewards[t])
        # 返回包含 probabilities、states 的字典。
        return {
            "probabilities": np.array(probabilities),
            "states": np.array(states),
        }

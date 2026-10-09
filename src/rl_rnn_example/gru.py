"""第四步：用 GRU 预测猴子的选择。

本框架专门处理一步反转学习，输入只有 [action, reward] 两项。
不再保留作者框架中与 action 重复的 stage2 字段。
forward 用于训练，必须保留梯度；predict_block 用于评估和查看状态。
"""

import numpy as np
import torch
from torch import nn


class GRUModel(nn.Module):
    """输入维度为 2，默认隐藏状态为 2 维，输出为两个动作的分数。"""

    def __init__(self, hidden_dim: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        # TODO 1：创建 GRU 层，输入维度为 2，隐藏维度为 hidden_dim。
        # 可以使用 nn.GRU；整个项目统一采用时间在前的维度顺序 (T, B, 特征数)。
        self.gru = nn.GRU(input_size=2, hidden_size=hidden_dim, batch_first=False)
        # TODO 2：创建线性输出层，把 hidden_dim 个状态映射为 2 个动作分数。
        # 初始隐藏状态固定为零，不定义成需要拟合的参数。
        self.linear = nn.Linear(hidden_dim, 2)
        self.h0 = None

    def forward(self, inputs: torch.Tensor) -> dict:
        """处理一批 block，返回保留梯度的 Tensor。

        输入：inputs 的形状为 (T, B, 2)，最后两项为 [action, reward]。
        T 是试次数，B 是 block 数；各 block 在时间上彼此独立。
        返回：
            logits: 形状为 (T, B, 2)，用于预测当前试次的动作分数。
            states: 形状为 (T, B, hidden_dim)，预测当前试次时的隐藏状态。

        第 t 行输出只能使用第 t 次之前的信息，不能先读 inputs[t] 再预测 actions[t]。
        """
        # TODO 1：为每个 block 建立独立的零初始状态，设备和类型与 inputs 一致。
        self.h0 = torch.zeros(
            1, inputs.size(1), self.hidden_dim, device=inputs.device, dtype=inputs.dtype
        )
        # TODO 2：运行 GRU，得到逐次读取输入后的状态。
        states, _ = self.gru(inputs, self.h0)
        # TODO 3：在状态序列开头放入初始零状态，去掉最后一个更新后的状态。
        # 这样第一条预测使用 h0，第二条预测使用读完第一条输入后的 h1。
        states = torch.cat([self.h0, states[:-1]], dim=0)
        # TODO 4：用输出层将对齐后的 states 转换成 logits，返回字典。
        # 训练时不要在这里转成 NumPy 或 detach，否则会断开梯度。
        # CrossEntropyLoss 直接接收 logits，不要提前对它执行 softmax。
        logits = self.linear(states)
        return {"logits": logits, "states": states}

    def predict_block(self, actions: np.ndarray, rewards: np.ndarray) -> dict:
        """评估一个 block，与 RWModel 使用相同的返回接口。

        输入：长度相同的一维 actions、rewards 数组。
        返回：NumPy 格式的 probabilities (T, 2) 和 states (T, hidden_dim)。
        states[t] 表示读取本次事件之前的隐藏状态，不直接等同于动作价值。
        """
        # TODO 1：将动作和奖励组成 (T, 1, 2) 的浮点 Tensor。
        # Tensor 的 dtype 和 device 与模型参数一致，避免 NumPy 默认的 float64 类型不匹配。
        inputs = torch.tensor(
            np.stack([actions, rewards], axis=-1), dtype=torch.float32
        ).unsqueeze(1)  # (T, 1, 2)
        # TODO 2：切换到评估模式，在 torch.no_grad() 中调用 forward。
        self.eval()
        with torch.no_grad():
            result = self.forward(inputs)
        logits = result["logits"]
        states = result["states"]
        # TODO 3：对 logits 的最后一维执行 softmax，得到两个动作的概率。
        probabilities = torch.softmax(logits, dim=-1)
        # TODO 4：去掉大小为 1 的 block 维，再转到 CPU 并转换为 NumPy。
        # 这个方法用于评估；训练必须调用 forward，以保留梯度。
        probabilities = probabilities.squeeze(1).cpu().numpy()
        states = states.squeeze(1).cpu().numpy()
        return {"probabilities": probabilities, "states": states}

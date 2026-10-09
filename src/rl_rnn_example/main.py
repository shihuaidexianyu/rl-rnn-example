"""实验入口：直接修改下面的配置，不使用命令行参数解析。

这里串联各模块的接口；数据处理和模型算法在对应模块中实现。
"""

from pathlib import Path

import torch

from .dataset import load_blocks
from .gru import GRUModel
from .metrics import evaluate
from .rw import RWModel
from .training import fit_gru, fit_rw, split_blocks


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"

# 一次只拟合一只猴子，改成 "W" 可以切换个体。
ANIMAL_NAME = "V"
TRIAL_START = 11
TRIAL_END = 70
SEED = 0

# 第一版采用一次固定划分，不进行完整的嵌套交叉验证。
VALIDATION_FRACTION = 0.1
TEST_FRACTION = 0.1

HIDDEN_DIM = 2
LEARNING_RATE = 0.005
MAX_EPOCHS = 1000
PATIENCE = 100


def main() -> None:
    """读取数据、划分 block、拟合模型，再用同一测试集进行比较。"""
    blocks = load_blocks(DATA_DIR, ANIMAL_NAME, TRIAL_START, TRIAL_END)
    train_blocks, validation_blocks, test_blocks = split_blocks(
        blocks,
        validation_fraction=VALIDATION_FRACTION,
        test_fraction=TEST_FRACTION,
        seed=SEED,
    )

    # 两个模型共享同一次数据划分，使用相同的训练和测试 block。
    rw_model = RWModel()
    fit_rw(rw_model, train_blocks)

    torch.manual_seed(SEED)
    gru_model = GRUModel(hidden_dim=HIDDEN_DIM)
    fit_gru(
        gru_model,
        train_blocks,
        validation_blocks,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
    )

    rw_result = evaluate(rw_model, test_blocks)
    gru_result = evaluate(gru_model, test_blocks)
    print("RW 测试表现：", rw_result)
    print("GRU 测试表现：", gru_result)


if __name__ == "__main__":
    main()

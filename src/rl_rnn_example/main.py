"""实验入口：直接修改下面的配置，不使用命令行参数解析。

这里串联各模块的接口；数据处理和模型算法在对应模块中实现。
"""

from pathlib import Path

import numpy as np
import torch

from .analysis import collect_predictions, compute_vector_field, load_results, save_results
from .dataset import load_blocks
from .gru import GRUModel
from .metrics import evaluate
from .plotting import (
    plot_behavior,
    plot_predictions,
    plot_state_trajectories,
    plot_training_history,
    plot_vector_field,
)
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

# True：拟合并保存结果；False：读取已有结果，直接重新画图。
# 同一输出目录再次拟合会更新该目录中的模型、数据和图片。
FIT_MODELS = True
OUTPUT_DIR = PROJECT_DIR / "results" / f"monkey_{ANIMAL_NAME}_seed{SEED}"

# 固定展示测试集中的第几个 block，不按模型表现挑选例子。
PLOT_BLOCK_INDEX = 0
GRID_SIZE = 13


def main() -> None:
    """拟合或加载模型，然后生成同一测试 block 的教材图片。"""
    if FIT_MODELS:
        if HIDDEN_DIM != 2:
            raise ValueError("当前二维动力学绘图使用两个隐藏单元，请设置 HIDDEN_DIM=2。")
        blocks = load_blocks(DATA_DIR, ANIMAL_NAME, TRIAL_START, TRIAL_END)
        train_blocks, validation_blocks, test_blocks = split_blocks(
            blocks,
            validation_fraction=VALIDATION_FRACTION,
            test_fraction=TEST_FRACTION,
            seed=SEED,
        )
        if not 0 <= PLOT_BLOCK_INDEX < len(test_blocks):
            raise ValueError("PLOT_BLOCK_INDEX 超出了测试 block 的范围。")
        print(
            f"block 数：训练 {len(train_blocks)}，验证 {len(validation_blocks)}，"
            f"测试 {len(test_blocks)}。",
            flush=True,
        )

        # 两个模型共享同一次数据划分，使用相同的训练和测试 block。
        print("正在拟合 RW……", flush=True)
        rw_model = RWModel()
        fit_rw(rw_model, train_blocks)

        print("正在拟合 GRU……", flush=True)
        torch.manual_seed(SEED)
        gru_model = GRUModel(hidden_dim=HIDDEN_DIM)
        history = fit_gru(
            gru_model,
            train_blocks,
            validation_blocks,
            learning_rate=LEARNING_RATE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
        )
        models = {"RW": rw_model, "GRU": gru_model}
        metrics = {}
        for model_name, model in models.items():
            metrics[model_name] = evaluate(model, test_blocks)
        records = collect_predictions(models, test_blocks)

        config = {
            "animal_name": ANIMAL_NAME,
            "trial_start": TRIAL_START,
            "trial_end": TRIAL_END,
            "seed": SEED,
            "validation_fraction": VALIDATION_FRACTION,
            "test_fraction": TEST_FRACTION,
            "hidden_dim": HIDDEN_DIM,
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "plot_block_index": PLOT_BLOCK_INDEX,
            "grid_size": GRID_SIZE,
        }
        splits = {
            "train": train_blocks,
            "validation": validation_blocks,
            "test": test_blocks,
        }
        # 先保存，再绘图。修改图形样式后可以复用这次拟合的结果。
        save_results(OUTPUT_DIR, models, records, config, splits, metrics, history)
    else:
        models, records, experiment = load_results(OUTPUT_DIR)
        metrics = experiment["metrics"]
        history = experiment["training_history"]

    for model_name, result in metrics.items():
        print(f"{model_name} 测试表现：{result}", flush=True)
    if not 0 <= PLOT_BLOCK_INDEX < len(records):
        raise ValueError("PLOT_BLOCK_INDEX 超出了保存的测试 block 范围。")
    record = records[PLOT_BLOCK_INDEX]
    print(
        f"绘图使用 {record['session_name']} 的 block {record['block_order']}，"
        f"原始第 {record['reversal_trial']} 次反转。",
        flush=True,
    )

    figure_dir = OUTPUT_DIR / "figures"
    plot_behavior(record, figure_dir)
    plot_predictions(record, figure_dir)
    plot_state_trajectories(record, figure_dir)
    plot_training_history(history, figure_dir)
    for model_name, model in models.items():
        prediction = record["models"][model_name]
        # T 个预测前状态加上末状态，组成包含全部 T 次更新的完整轨迹。
        full_states = np.vstack([prediction["states"], prediction["final_state"]])
        state_limits = (0.0, 1.0) if model_name == "RW" else (-1.0, 1.0)
        field = compute_vector_field(model, full_states, GRID_SIZE, state_limits)
        plot_vector_field(field, model_name, record, figure_dir)
    print(f"结果保存至：{OUTPUT_DIR}", flush=True)
    print(f"PNG 和 SVG 图片保存至：{figure_dir}", flush=True)


if __name__ == "__main__":
    main()

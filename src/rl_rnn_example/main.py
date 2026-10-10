"""实验入口：直接修改下面的配置，不使用命令行参数解析。

这里串联各模块的接口；数据处理和模型算法在对应模块中实现。
"""

from pathlib import Path

import torch

from .analysis import (
    collect_predictions,
    compute_model_dynamics,
    compute_one_step,
    load_results,
    save_figure_captions,
    save_results,
    summarize_model_comparison,
)
from .dataset import load_blocks
from .gru import GRUModel
from .metrics import evaluate
from .plotting import (
    plot_behavior,
    plot_block_example,
    plot_dynamics_comparison,
    plot_model_comparison,
    plot_one_step,
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
FIT_MODELS = False
OUTPUT_DIR = PROJECT_DIR / "results" / f"monkey_{ANIMAL_NAME}_seed{SEED}"

# 主图汇总全部测试 block，不由下面的单个 block 编号决定。
# 网格用于计算固定模型在测试集状态范围内的更新幅度与等值线。
GRID_SIZE = 41
# None：显示整个测试集的全部更新；设正整数时，每种输入至多抽取这么多支箭头。
# 两个模型使用相同种子，确保抽取的是同一批真实事件。
MAX_ARROWS_PER_CONDITION = None
DYNAMICS_SAMPLE_SEED = 0

# 以下仅控制可选补充图，默认不生成单个 block 和单次更新的示例。
PLOT_SUPPLEMENTARY = False
PLOT_BLOCK_INDEX = 0
# 使用 block 中的原始试次编号，不是数组下标。
ONE_STEP_TRIAL = 20
# 两项都为 None 时读取该试次的实际动作和奖励。
# 例如同时改成 0、1，可以问“从这个状态选择动作 0 并得到奖励，会怎样”。
PROBE_ACTION = None
PROBE_REWARD = None
# None：两个模型都使用该试次前的真实行为历史所驱动的状态。
# 也可以改成 {"RW": [0.6, 0.2], "GRU": [-0.5, 0.3]}，探查自己指定的状态。
PROBE_STATES = None


def main() -> None:
    """比较测试预测，并汇总整个测试集，绘制固定模型的条件动力学。"""
    if FIT_MODELS:
        blocks = load_blocks(DATA_DIR, ANIMAL_NAME, TRIAL_START, TRIAL_END)
        train_blocks, validation_blocks, test_blocks = split_blocks(
            blocks,
            validation_fraction=VALIDATION_FRACTION,
            test_fraction=TEST_FRACTION,
            seed=SEED,
        )
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
            "max_arrows_per_condition": MAX_ARROWS_PER_CONDITION,
            "dynamics_sample_seed": DYNAMICS_SAMPLE_SEED,
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
    figure_dir = OUTPUT_DIR / "figures"
    # 图一：在同一测试集上预测猴子的选择，评价当前这两套已拟合参数。
    comparison = summarize_model_comparison(records)
    plot_model_comparison(comparison, figure_dir)

    # 图二：每个 block 内先独立计算更新，再按四种输入汇总全部测试事件。
    # 两行分别是 RW 和 GRU；每列固定一个动作与奖励对。
    print(f"模型动力学图汇总全部 {len(records)} 个测试 block。", flush=True)
    fields = {}
    for model_name, model in models.items():
        state_limits = (0.0, 1.0) if model_name == "RW" else (-1.0, 1.0)
        fields[model_name] = compute_model_dynamics(
            model,
            records,
            model_name,
            grid_size=GRID_SIZE,
            state_limits=state_limits,
            max_arrows_per_condition=MAX_ARROWS_PER_CONDITION,
            sample_seed=DYNAMICS_SAMPLE_SEED,
        )
    plot_dynamics_comparison(fields, figure_dir)

    # 单次更新仍可用于辅助讲解，但不作为默认的动力学主图。
    step = None
    if PLOT_SUPPLEMENTARY:
        record = records[PLOT_BLOCK_INDEX]
        step = compute_one_step(
            models,
            record,
            ONE_STEP_TRIAL,
            action=PROBE_ACTION,
            reward=PROBE_REWARD,
            initial_states=PROBE_STATES,
        )
        plot_one_step(step, figure_dir)
        plot_block_example(record, figure_dir)
        plot_behavior(record, figure_dir)
        plot_predictions(record, figure_dir)
        plot_state_trajectories(record, figure_dir)
        plot_training_history(history, figure_dir)
        for model_name, field in fields.items():
            plot_vector_field(field, model_name, figure_dir)
    save_figure_captions(comparison, fields, figure_dir, step=step)
    print(f"结果保存至：{OUTPUT_DIR}", flush=True)
    print(f"PNG 和 SVG 图片保存至：{figure_dir}", flush=True)


if __name__ == "__main__":
    main()

"""整理逐试次预测、计算状态更新，并保存可以重复分析的实验结果。

这里只调用模型已有的接口，不拟合参数。模型拟合与绘图分别在
training.py 和 plotting.py，便于单独查看每一步在做什么。
"""

import json
from pathlib import Path

import numpy as np
import torch

from .gru import GRUModel
from .rw import RWModel


def collect_predictions(models: dict, blocks: list[dict]) -> list[dict]:
    """逐个独立 block 预测，保留元信息、选择概率和状态。

    返回记录中的 models 字段，按 RW、GRU 分别保存各自的预测结果。
    states[t] 是读取第 t 次反馈前的状态，final_state 是最后一次反馈后状态。
    """
    records = []
    for block in blocks:
        record = block.copy()
        record["models"] = {}
        for model_name, model in models.items():
            prediction = model.predict_block(block["actions"], block["rewards"])
            record["models"][model_name] = prediction
        records.append(record)
    return records


def compute_vector_field(
    model,
    states: np.ndarray,
    grid_size: int = 17,
    state_limits: tuple[float, float] | None = None,
) -> dict:
    """在真实轨迹附近计算四种动作、奖励条件下的一步状态变化。

    states 为 (T+1, 2)，包含起点和最后一次更新后的终点。
    state_limits 可设为 RW 的 (0, 1) 或当前 GRU 的 (-1, 1)。
    返回的 dx、dy 是原始一步变化量，不把箭头统一成相同长度。
    网格点是用于探查模型的假设状态，不表示猴子实际访问过这些点。
    """
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 2 or len(states) == 0:
        raise ValueError("二维动力学图需要形状为 (试次数, 2) 的非空状态数组。")
    if not np.isfinite(states).all():
        raise ValueError("状态中不能包含缺失值或无穷大。")
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("每个坐标轴至少需要两个网格点。")

    # 各方向留一点边距；几乎不变的状态也应有足够的可视范围。
    lower = states.min(axis=0)
    upper = states.max(axis=0)
    padding = np.maximum((upper - lower) * 0.12, 0.05)
    lower = lower - padding
    upper = upper + padding
    if state_limits is not None:
        lower = np.maximum(lower, state_limits[0])
        upper = np.minimum(upper, state_limits[1])
    if np.any(lower >= upper):
        raise ValueError("状态范围与给定的坐标边界不相容。")

    x_values = np.linspace(lower[0], upper[0], grid_size)
    y_values = np.linspace(lower[1], upper[1], grid_size)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    # 背景色表示更新前选择动作 0 的概率，两种模型都使用 0–1 概率刻度。
    probabilities = []
    for state in points:
        probabilities.append(model.choice_probabilities(state)[0])
    probability_grid = np.asarray(probabilities).reshape(x_grid.shape)

    conditions = []
    for action in (0, 1):
        for reward in (0, 1):
            changes = []
            for state in points:
                next_state = model.update_state(state, action, reward)
                changes.append(next_state - state)
            changes = np.asarray(changes)
            conditions.append(
                {
                    "action": action,
                    "reward": reward,
                    "dx": changes[:, 0].reshape(x_grid.shape),
                    "dy": changes[:, 1].reshape(y_grid.shape),
                }
            )

    return {
        "x": x_grid,
        "y": y_grid,
        "probability": probability_grid,
        "conditions": conditions,
        "observed_states": states.copy(),
    }


def save_results(
    output_dir: Path,
    models: dict,
    records: list[dict],
    config: dict,
    splits: dict,
    metrics: dict,
    history: dict,
) -> None:
    """保存模型、测试集逐试次预测、配置、数据划分和训练记录。

    JSON 文件可直接阅读；GRU 权重单独使用 PyTorch 的 state_dict 保存。
    保存数据划分的身份信息，避免后续把不同实验的模型和 block 混在一起。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_ids = {}
    for split_name, blocks in splits.items():
        split_ids[split_name] = []
        for block in blocks:
            split_ids[split_name].append(
                {
                    "animal_name": block["animal_name"],
                    "session_name": block["session_name"],
                    "block_order": block["block_order"],
                }
            )

    experiment = {
        "config": config,
        "splits": split_ids,
        "metrics": metrics,
        "training_history": history,
        "models": {
            "RW": {
                "alpha": float(models["RW"].alpha),
                "iTemp": float(models["RW"].iTemp),
            },
            "GRU": {"hidden_dim": models["GRU"].hidden_dim},
        },
    }
    with (output_dir / "experiment.json").open("w", encoding="utf-8") as file:
        json.dump(experiment, file, ensure_ascii=False, indent=2, allow_nan=False)

    # 显式地把数组转换成列表，保存格式与内存里的字典保持一一对应。
    saved_records = []
    for record in records:
        saved_record = record.copy()
        for key in ("trial_numbers", "actions", "rewards"):
            saved_record[key] = record[key].tolist()
        saved_record["models"] = {}
        for model_name, prediction in record["models"].items():
            saved_record["models"][model_name] = {}
            for key in ("probabilities", "states", "final_state"):
                saved_record["models"][model_name][key] = prediction[key].tolist()
        saved_records.append(saved_record)
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as file:
        json.dump(saved_records, file, ensure_ascii=False, indent=2, allow_nan=False)

    torch.save(models["GRU"].state_dict(), output_dir / "gru_weights.pt")


def load_results(output_dir: Path) -> tuple[dict, list[dict], dict]:
    """读取已经保存的实验，返回 models、records、experiment，用于直接重画。"""
    output_dir = Path(output_dir)
    with (output_dir / "experiment.json").open(encoding="utf-8") as file:
        experiment = json.load(file)
    with (output_dir / "predictions.json").open(encoding="utf-8") as file:
        records = json.load(file)

    rw_parameters = experiment["models"]["RW"]
    rw_model = RWModel(alpha=rw_parameters["alpha"], iTemp=rw_parameters["iTemp"])
    gru_model = GRUModel(hidden_dim=experiment["models"]["GRU"]["hidden_dim"])
    weights = torch.load(output_dir / "gru_weights.pt", map_location="cpu", weights_only=True)
    gru_model = gru_model.to(dtype=weights["linear.weight"].dtype)
    gru_model.load_state_dict(weights)
    gru_model.eval()

    for record in records:
        for key in ("trial_numbers", "actions", "rewards"):
            record[key] = np.asarray(record[key], dtype=int)
        for prediction in record["models"].values():
            for key in ("probabilities", "states", "final_state"):
                prediction[key] = np.asarray(prediction[key], dtype=float)
    return {"RW": rw_model, "GRU": gru_model}, records, experiment

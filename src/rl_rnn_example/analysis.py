"""整理逐试次预测、计算状态更新，并保存可以重复分析的实验结果。

这里只调用模型已有的接口，不拟合参数。模型拟合与绘图分别在
training.py 和 plotting.py，便于单独查看每一步在做什么。
"""

import json
from pathlib import Path

import numpy as np
import torch

from .gru import GRUModel
from .metrics import accuracy, negative_log_likelihood
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


def summarize_model_comparison(records: list[dict]) -> dict:
    """用同一批测试试次比较 RW 和 GRU，不重新拟合或选择模型。

    records 必须来自测试集。这里汇总已经独立预测的各 block 的概率，
    每次试次占相同权重；不是先算各 block 的平均分，再把 block 等权平均。
    准确率衡量是否预测到猴子的选择，不是是否选中了高奖励概率选项。
    一次固定划分的结果不能代替跨划分检验，也不代表所有经典模型的表现。
    """
    if not records:
        raise ValueError("模型比较至少需要一个测试 block。")
    animals = {record["animal_name"] for record in records}
    if len(animals) != 1:
        raise ValueError("一次模型比较只使用同一只猴子的测试记录。")

    actions = np.concatenate([record["actions"] for record in records])
    if len(actions) == 0:
        raise ValueError("测试记录中至少需要一次试次。")
    comparison = {
        "animal_name": records[0]["animal_name"],
        "n_blocks": len(records),
        "n_sessions": len({record["session_name"] for record in records}),
        "n_trials": len(actions),
        "models": {},
    }
    for model_name in ("RW", "GRU"):
        block_probabilities = []
        for record in records:
            probabilities = np.asarray(record["models"][model_name]["probabilities"])
            if probabilities.shape != (len(record["actions"]), 2):
                raise ValueError("每次真实选择必须对应两个动作的预测概率。")
            block_probabilities.append(probabilities)
        # 拼接的是已经算好的预测，不会把一个 block 的记忆带入下一个 block。
        probabilities = np.concatenate(block_probabilities)
        comparison["models"][model_name] = {
            "nll": float(negative_log_likelihood(probabilities, actions)),
            "accuracy": float(accuracy(probabilities, actions)),
        }
    return comparison


def compute_one_step(
    models: dict,
    record: dict,
    trial_number: int,
    action: int | None = None,
    reward: int | None = None,
    initial_states: dict | None = None,
) -> dict:
    """给两个固定模型相同的动作和奖励，查看各自的状态如何更新一步。

    默认取指定原始试次的事件前状态，即两个模型已读过相同的行为历史。
    action、reward 都省略时使用该试次的实际输入；也可以同时指定它们，
    提问“如果此刻发生另一种反馈，会怎样”。这不会改写原有行为序列。
    initial_states 可以写成 {"RW": [0.6, 0.2], "GRU": [-0.5, 0.3]}，
    用于探查手动设定的 Q pair 和 h pair；两套坐标不具有相同含义。
    返回值同时包含更新前后状态和选择概率，不修改权重或已保存的状态。
    """
    trial_numbers = np.asarray(record["trial_numbers"])
    matches = np.flatnonzero(trial_numbers == trial_number)
    if len(matches) != 1:
        raise ValueError("请使用该 block 保留范围内唯一的原始试次编号。")
    trial_index = int(matches[0])
    observed_action = int(record["actions"][trial_index])
    observed_reward = int(record["rewards"][trial_index])

    if (action is None) != (reward is None):
        raise ValueError("动作和奖励必须一起指定，或一起省略以使用实际输入。")
    if action is None:
        action = observed_action
        reward = observed_reward
    if action not in (0, 1) or reward not in (0, 1):
        raise ValueError("动作和奖励只接受 0、1 编码。")
    if initial_states is not None and set(initial_states) != set(models):
        raise ValueError("手动探查时，请为每个模型分别提供一个初始状态。")

    source = {
        "animal_name": record["animal_name"],
        "session_name": record["session_name"],
        "block_order": record["block_order"],
        "block_type": record["block_type"],
        "trial_number": int(trial_number),
        "state_source": "history" if initial_states is None else "manual",
        "observed_action": observed_action,
        "observed_reward": observed_reward,
    }
    step = {
        "source": source,
        "action": int(action),
        "reward": int(reward),
        "is_observed_transition": (
            initial_states is None and action == observed_action and reward == observed_reward
        ),
        "models": {},
    }
    for model_name, model in models.items():
        if initial_states is None:
            before = record["models"][model_name]["states"][trial_index]
        else:
            before = initial_states[model_name]
        before = np.array(before, dtype=float, copy=True)
        if before.shape != (2,) or not np.isfinite(before).all():
            raise ValueError("一步示意图需要每个模型提供两个有限数值组成的状态。")

        # 直接从当前状态读取这次输入；不要调用会从零重置的 predict_block。
        after = model.update_state(before, int(action), int(reward))
        step["models"][model_name] = {
            "before": before,
            "after": after,
            "change": after - before,
            "probabilities_before": model.choice_probabilities(before),
            "probabilities_after": model.choice_probabilities(after),
        }
    return step


def save_figure_captions(
    comparison: dict,
    fields: dict,
    output_dir: str | Path,
    step: dict | None = None,
) -> None:
    """把主图的说明另存为图注，图内只保留读图所需的标记。

    图注由本次分析的数据生成；切换被试、试次或假设输入后，说明也同步更新。
    不改动原先保存的模型和实验结果。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_caption = (
        "图 1 | RW 与 GRU 的测试集预测表现。"
        "a，每试次平均负对数似然（NLL，越低越好）；"
        "b，选择预测准确率（越高越好）。"
        f"两个模型使用猴子 {comparison['animal_name']} 的同一测试集，"
        f"包含 {comparison['n_sessions']} 个 session、{comparison['n_blocks']} 个 block、"
        f"{comparison['n_trials']} 次试次，指标按试次数加权。"
        "预测目标为猴子的实际选择。图中为单被试、单次固定划分的描述性结果，"
        "未进行统计推断。\n"
    )
    (output_dir / "model_comparison.caption.txt").write_text(comparison_caption, encoding="utf-8")

    # 模型更新规则由固定参数决定；测试集决定实际访问的状态和事件分布。
    # 元信息读取本次汇总，而不是沿用拟合时保存的旧绘图配置。
    field = fields["RW"]
    source_blocks = field["source_blocks"]
    n_sessions = len({block["session_name"] for block in source_blocks})
    condition_order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    condition_lookup = {
        (condition["action"], condition["reward"]): condition
        for condition in field["conditions"]
    }
    event_counts = [str(condition_lookup[key]["n_observed"]) for key in condition_order]
    display_descriptions = []
    for model_name, model_field in fields.items():
        n_displayed = sum(condition["n_displayed"] for condition in model_field["conditions"])
        display_descriptions.append(f"{model_name} {n_displayed} 次")
    sampling_note = "显示全部更新。"
    if field["max_arrows_per_condition"] is not None:
        sampling_note = (
            f"每种输入最多抽取 {field['max_arrows_per_condition']} 次更新，"
            f"固定抽样种子 {field['sample_seed']}，RW 与 GRU 取相同事件。"
        )
    dynamics_caption = (
        "图 2 | 整个测试集覆盖状态空间中的条件动力学。"
        "上排为 RW（a–d），下排为 GRU（e–h）；"
        "四列分别固定动作与奖励 (a_t,r_t)=(0,0)、(0,1)、(1,0)、(1,1)。"
        f"使用猴子 {source_blocks[0]['animal_name']} 的全部测试记录："
        f"{n_sessions} 个 session、{len(source_blocks)} 个 block、{field['n_trials']} 次反馈；"
        f"四种条件分别包含 {'、'.join(event_counts)} 次事件。"
        "黑箭头表示真实行为驱动的模型状态更新 s_t→s_{t+1}，"
        f"图中显示 {'、'.join(display_descriptions)}。"
        + sampling_note
        + "背景色及细等值线表示给定该列输入后，网格状态的更新幅度 ||s_{t+1}−s_t||₂；"
        "网格范围由整个测试集的状态（含各 block 末状态）确定，"
        "网格点不一定被实际访问，等值线不是时间轨迹。"
        "同一模型四个面板共用色标；Q 与 h 的坐标单位不同，不能跨模型比较更新幅度。"
        "橙色虚线表示两个动作等概率的边界，即 P(a=0)=0.5。"
        "各 block 独立从零初始化，反转时不重置；更新先在 block 内计算，再汇总，"
        "不连接不同 block 的首尾。两个模型各自使用一套固定参数。"
        "动作 0/1 在 what block 中表示图像，在 where block 中表示位置；h 为模型隐藏状态。\n"
    )
    (output_dir / "dynamics_comparison.caption.txt").write_text(dynamics_caption, encoding="utf-8")

    if step is None:
        return

    source = step["source"]
    reference = (
        f"猴子 {source['animal_name']}，{source['session_name']}，"
        f"block {source['block_order']}（{source['block_type']}），"
        f"原始试次 {source['trial_number']}"
    )
    if source["state_source"] == "manual":
        source_description = f"以{reference}的动作编码为参照，起点为手动指定的假设状态。"
    elif step["is_observed_transition"]:
        source_description = (
            f"示例取自{reference}。两个起点由此前相同的保留行为历史驱动模型产生，"
            "输入为该试次的实际动作与奖励。"
        )
    else:
        source_description = (
            f"起点取自{reference}输入前的模型状态，"
            "本图替换该试次的动作与奖励进行假设探查。"
        )
    action_type = "图像" if source["block_type"] == "what" else "位置"
    reward_description = "获得奖励" if step["reward"] == 1 else "未获奖励"
    step_caption = (
        "补充图 | 相同输入下的单步状态更新。a，RW；b，GRU。"
        f"输入为选择{action_type} {step['action']}（a_t={step['action']}），"
        f"{reward_description}（r_t={step['reward']}）。"
        "空心圆表示输入前的状态 s_t，实心圆表示更新后的状态 s_{t+1}；"
        "箭头连接两者，零位移时两点重合。"
        + source_description
        + "下方数字为更新前后的动作 0 选择概率，后者用于下一次选择。"
        "Q_0、Q_1 表示选项价值，h_1、h_2 为隐藏状态坐标；"
        "各面板内部采用等比例坐标，两个模型的坐标含义和范围不同。模型参数固定。\n"
    )
    (output_dir / "one_step.caption.txt").write_text(step_caption, encoding="utf-8")


def get_readout(model) -> tuple[np.ndarray, float]:
    """返回动作 0 相对动作 1 的分数差：w · state + b。

    分数差越大，选择动作 0 的概率越大；分数差为零时两个动作等概率。
    直接读取已拟合参数即可，不需要像原作者的通用接口那样再做一次回归。
    这里保留权重原始大小，画读出方向时才把向量转为单位方向。
    """
    if isinstance(model, RWModel):
        weights = np.array([model.iTemp, -model.iTemp], dtype=float)
        bias = 0.0
    elif isinstance(model, GRUModel):
        weights = model.linear.weight.detach().cpu().numpy()
        biases = model.linear.bias.detach().cpu().numpy()
        weights = np.asarray(weights[0] - weights[1], dtype=float)
        bias = float(biases[0] - biases[1])
    else:
        raise TypeError("读出分析只支持本项目的 RWModel 和 GRUModel。")
    return weights, bias


def compute_vector_field(
    model,
    states: np.ndarray,
    grid_size: int = 17,
    state_limits: tuple[float, float] | None = None,
    actions: np.ndarray | None = None,
    rewards: np.ndarray | None = None,
) -> dict:
    """按原文图 4a 的含义，计算更新幅度、真实更新箭头和读出方向。

    states 中的二维坐标用于确定背景网格的范围。
    提供 actions、rewards 时，states 必须是单个 block 的 (T+1, 2) 状态，
    事件数组长度为 T。多个 block 请调用 compute_model_dynamics，避免跨块求差。
    颜色与等值线使用网格上的一步更新幅度 sqrt(dx² + dy²)。
    state_limits 可设为 RW 的 (0, 1) 或当前 GRU 的 (-1, 1)。
    未提供真实事件时仍可探查网格，但图中的箭头必须标为假设状态更新。
    网格点不表示猴子实际访问过的位置；模型状态也不等于记录到的神经活动。
    """
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 2 or len(states) < 2:
        raise ValueError("动力学图需要形状为 (T+1, 2) 的二维状态序列。")
    if not np.isfinite(states).all():
        raise ValueError("状态中不能包含非有限数值。")
    if not isinstance(grid_size, (int, np.integer)) or grid_size < 2:
        raise ValueError("网格每个方向至少需要 2 个点。")
    readout_vector, readout_bias = get_readout(model)
    if readout_vector.shape != (2,):
        raise ValueError("二维动力学图只支持具有两个内部状态的模型。")

    has_observed_events = actions is not None and rewards is not None
    if (actions is None) != (rewards is None):
        raise ValueError("动作和奖励必须一起提供。")
    if has_observed_events:
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        expected_shape = (len(states) - 1,)
        if actions.shape != expected_shape or rewards.shape != expected_shape:
            raise ValueError("T 次动作和奖励必须对应 T+1 个状态。")
        if not np.isin(actions, [0, 1]).all() or not np.isin(rewards, [0, 1]).all():
            raise ValueError("动作和奖励只接受 0、1 编码。")
        observed_changes = states[1:] - states[:-1]

    # 各方向留一点边距；几乎不变的状态也应有足够的可视范围。
    lower = states.min(axis=0)
    upper = states.max(axis=0)
    padding = np.maximum((upper - lower) * 0.12, 0.05)
    lower = lower - padding
    upper = upper + padding
    if state_limits is not None:
        lower = np.maximum(lower, state_limits[0])
        upper = np.minimum(upper, state_limits[1])

    x_values = np.linspace(lower[0], upper[0], grid_size)
    y_values = np.linspace(lower[1], upper[1], grid_size)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    # 保留概率供其他分析使用；原文图 4a 的背景色使用更新幅度，不使用概率。
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
            speed = np.linalg.norm(changes, axis=1).reshape(x_grid.shape)

            # 原图的黑箭头来自真实行为驱动的模型状态，并按本次事件筛选。
            # 不将整个 block 的混合事件轨迹重复放进每一个固定条件面板。
            condition_states = np.empty((0, 2))
            condition_changes = np.empty((0, 2))
            if has_observed_events:
                selected = (actions == action) & (rewards == reward)
                condition_states = states[:-1][selected].copy()
                condition_changes = observed_changes[selected].copy()
            conditions.append(
                {
                    "action": action,
                    "reward": reward,
                    "dx": changes[:, 0].reshape(x_grid.shape),
                    "dy": changes[:, 1].reshape(y_grid.shape),
                    "speed": speed,
                    "observed_states": condition_states,
                    "observed_changes": condition_changes,
                }
            )

    # 四种输入共享一个幅度上限，因此同一模型的四个面板可以比较颜色。
    # RW 与 GRU 的状态坐标和单位不同，不据此比较两种模型的学习率。
    speed_max = max(float(condition["speed"].max()) for condition in conditions)
    return {
        "x": x_grid,
        "y": y_grid,
        "probability": probability_grid,
        "conditions": conditions,
        "observed_states": states.copy(),
        "has_observed_events": has_observed_events,
        "speed_max": speed_max,
        "readout_vector": readout_vector,
        "readout_bias": readout_bias,
    }


def compute_model_dynamics(
    model,
    records: list[dict],
    model_name: str,
    grid_size: int = 41,
    state_limits: tuple[float, float] | None = None,
    max_arrows_per_condition: int | None = 100,
    sample_seed: int = 0,
) -> dict:
    """固定一套模型权重，汇总同一只猴子多个 block 的动力学。

    records 使用 collect_predictions 或 load_results 返回的逐 block 记录。
    所有状态共同确定背景范围，每个 block 的更新必须先独立计算再汇总。
    箭头过多时只抽样显示，背景、等值线和读出方向仍使用完整模型计算。
    max_arrows_per_condition=None 表示显示全部箭头；固定种子便于重复绘图。
    """
    if not records:
        raise ValueError("汇总动力学至少需要一个 block。")
    animals = {record["animal_name"] for record in records}
    if len(animals) != 1:
        raise ValueError("同一张模型动力学图只汇总同一只猴子的 block。")
    if max_arrows_per_condition is not None and max_arrows_per_condition < 1:
        raise ValueError("每种条件的箭头上限应为正整数，或用 None 显示全部。")

    all_states = []
    update_starts = []
    update_changes = []
    all_actions = []
    all_rewards = []
    source_blocks = []
    for record in records:
        prediction = record["models"][model_name]
        states = np.asarray(prediction["states"], dtype=float)
        final_state = np.asarray(prediction["final_state"], dtype=float)
        actions = np.asarray(record["actions"])
        rewards = np.asarray(record["rewards"])
        if states.shape != (len(actions), 2) or rewards.shape != actions.shape:
            raise ValueError("每个 block 的二维状态、动作和奖励需要逐次对应。")

        # 预测时每个 block 都独立从零状态开始，这里读取其保存的完整轨迹。
        # 先在各 block 内计算差分，绝不能先拼接轨迹再求差：那会把前一个
        # block 的末状态到下一个 block 初状态的重置，误画成一次学习更新。
        complete_states = np.vstack([states, final_state])
        changes = complete_states[1:] - complete_states[:-1]
        all_states.append(complete_states)
        update_starts.append(states)
        update_changes.append(changes)
        all_actions.append(actions)
        all_rewards.append(rewards)
        source_blocks.append(
            {
                "animal_name": record["animal_name"],
                "session_name": record["session_name"],
                "block_order": record["block_order"],
                "block_type": record["block_type"],
            }
        )

    # 此处拼接只形成一组空间位置，用于确定网格范围，不作为连续时间序列。
    # 包括所有 block 的末状态，且在抽样箭头之前确定范围。
    field = compute_vector_field(
        model,
        np.concatenate(all_states),
        grid_size=grid_size,
        state_limits=state_limits,
    )
    starts = np.concatenate(update_starts)
    changes = np.concatenate(update_changes)
    actions = np.concatenate(all_actions)
    rewards = np.concatenate(all_rewards)
    rng = np.random.default_rng(sample_seed)
    for condition in field["conditions"]:
        selected = (actions == condition["action"]) & (rewards == condition["reward"])
        event_indices = np.flatnonzero(selected)
        n_observed = len(event_indices)
        if max_arrows_per_condition is not None and n_observed > max_arrows_per_condition:
            event_indices = np.sort(
                rng.choice(event_indices, size=max_arrows_per_condition, replace=False)
            )
        # 起点和变化量使用同一组索引；相同种子使 RW 与 GRU 展示相同事件。
        condition["observed_states"] = starts[event_indices].copy()
        condition["observed_changes"] = changes[event_indices].copy()
        condition["event_indices"] = event_indices
        condition["n_observed"] = n_observed
        condition["n_displayed"] = len(event_indices)

    field["has_observed_events"] = True
    field["source_blocks"] = source_blocks
    field["n_trials"] = len(actions)
    field["max_arrows_per_condition"] = max_arrows_per_condition
    field["sample_seed"] = sample_seed
    return field


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

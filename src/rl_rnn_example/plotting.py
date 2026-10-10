"""把固定模型的更新规则和单个 block 的运行过程画成教学图。

这里只接收已经计算好的数组，不拟合模型，也不改变模型状态。
每张图同时保存 PNG 和 SVG。若没有中文字体，则自动改用英文图表标签。
"""

from pathlib import Path

import matplotlib

# 所有图直接保存到文件，不弹出需要手动关闭的窗口。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import numpy as np


def _configure_fonts() -> bool:
    """优先选择常见中文字体；返回图表是否可以使用中文标签。"""
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
    ]
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["font.family"] = "DejaVu Sans"
    return False


USE_CHINESE = _configure_fonts()
MODEL_COLORS = {
    "RW": "#c97722",
    "GRU": "#2864ba",
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "#fcfcfd",
        "legend.framealpha": 0.9,
        "svg.fonttype": "none",
    }
)


def _label(chinese: str, english: str) -> str:
    if USE_CHINESE:
        return chinese
    return english


def _action_labels(block: dict) -> tuple[str, str]:
    """按数据中的 0/1 编码标注，避免未经核对便指定具体图像或左右侧。"""
    if block["block_type"] == "what":
        return _label("图像 0", "image 0"), _label("图像 1", "image 1")
    return _label("位置 0", "location 0"), _label("位置 1", "location 1")


def _block_title(block: dict) -> str:
    animal_name = block["animal_name"]
    session_name = block["session_name"]
    block_order = block["block_order"]
    block_type = block["block_type"]
    return _label(
        f"猴子 {animal_name} · {session_name} · block {block_order} · {block_type}",
        f"Monkey {animal_name} · {session_name} · block {block_order} · {block_type}",
    )


def _state_labels(model_name: str, block: dict) -> tuple[str, str]:
    if model_name == "RW":
        action_zero, action_one = _action_labels(block)
        return f"Q({action_zero})", f"Q({action_one})"
    # 隐藏坐标没有预先规定的认知含义，因此不把 GRU 的坐标称为 Q 值。
    return "h1", "h2"


def _trial_axis(axis, block: dict, show_reversal_label: bool = False) -> None:
    """横轴使用原始试次；只在当前保留范围内标出真实程序反转。"""
    trial_numbers = np.asarray(block["trial_numbers"])
    axis.set_xlim(trial_numbers[0] - 0.5, trial_numbers[-1] + 0.5)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
    reversal_trial = block.get("reversal_trial")
    if reversal_trial is None:
        return
    if trial_numbers[0] <= reversal_trial <= trial_numbers[-1]:
        reversal_label = None
        if show_reversal_label:
            reversal_label = _label("程序反转", "Programmed reversal")
        axis.axvline(
            reversal_trial,
            color="#a94646",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            label=reversal_label,
        )


def _save(figure, output_dir: str | Path, filename: str) -> None:
    """保存同一张图的两种格式，完成后释放图对象。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_dir / f"{filename}.png", dpi=180, bbox_inches="tight")
        figure.savefig(output_dir / f"{filename}.svg", bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_behavior(block: dict, output_dir: str | Path) -> None:
    """绘制真实动作和奖励，不包含模型预测。"""
    trials = np.asarray(block["trial_numbers"])
    actions = np.asarray(block["actions"])
    rewards = np.asarray(block["rewards"])
    action_zero, action_one = _action_labels(block)

    figure, axes = plt.subplots(2, 1, figsize=(10.5, 6), sharex=True)
    choice_axis = axes[0]
    reward_axis = axes[1]
    choice_axis.step(trials, actions, where="mid", color="#64748b", alpha=0.45)
    choice_axis.scatter(trials, actions, s=28, color="#334155", zorder=3)
    choice_axis.set_yticks([0, 1], [action_zero, action_one])
    choice_axis.set_ylim(-0.15, 1.15)
    choice_axis.set_ylabel(_label("实际选择", "Observed choice"))
    choice_axis.set_title(_label("猴子的选择记录", "Observed choices"))
    _trial_axis(choice_axis, block, show_reversal_label=True)
    if choice_axis.get_legend_handles_labels()[0]:
        choice_axis.legend(loc="best", fontsize=9)

    # 奖励是实际获得的反馈；未获奖励不意味着一定选错了高概率选项。
    reward_axis.vlines(trials, 0, rewards, color="#c97722", linewidth=1.5, alpha=0.65)
    reward_axis.scatter(trials, rewards, s=24, color="#c97722", zorder=3)
    reward_axis.set_yticks([0, 1], [_label("无奖励", "No reward"), _label("有奖励", "Reward")])
    reward_axis.set_ylim(-0.15, 1.15)
    reward_axis.set_ylabel(_label("实际反馈", "Observed outcome"))
    reward_axis.set_xlabel(_label("block 内原始试次编号", "Original trial within block"))
    _trial_axis(reward_axis, block)

    figure.suptitle(_block_title(block), fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    _save(figure, output_dir, "behavior")


def plot_predictions(record: dict, output_dir: str | Path) -> None:
    """模型曲线是事件前的 P(action=0)，圆点是是否实际选择了动作 0。"""
    trials = np.asarray(record["trial_numbers"])
    actions = np.asarray(record["actions"])
    rewards = np.asarray(record["rewards"])
    action_zero, action_one = _action_labels(record)
    observed_action_zero = (actions == 0).astype(float)

    figure, axis = plt.subplots(figsize=(10.5, 5))
    for model_name in ("RW", "GRU"):
        probabilities = np.asarray(record["models"][model_name]["probabilities"])
        axis.plot(
            trials,
            probabilities[:, 0],
            color=MODEL_COLORS[model_name],
            linewidth=2,
            label=model_name,
        )

    rewarded_trials = rewards == 1
    axis.scatter(
        trials[rewarded_trials],
        observed_action_zero[rewarded_trials],
        s=28,
        color="#334155",
        label=_label("实际选择，有奖励", "Observed choice, reward"),
        zorder=3,
    )
    unrewarded_trials = rewards == 0
    axis.scatter(
        trials[unrewarded_trials],
        observed_action_zero[unrewarded_trials],
        s=28,
        facecolors="white",
        edgecolors="#64748b",
        label=_label("实际选择，无奖励", "Observed choice, no reward"),
        zorder=3,
    )
    _trial_axis(axis, record, show_reversal_label=True)
    axis.set_ylim(-0.1, 1.1)
    axis.set_ylabel(_label(f"选择{action_zero}的概率", f"P(choose {action_zero})"))
    axis.set_xlabel(_label("block 内原始试次编号", "Original trial within block"))
    # 图例放在坐标区上方，避免遮挡概率为 1 附近的真实选择圆点。
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=3,
        fontsize=9,
    )
    figure.suptitle(_block_title(record), fontsize=13)
    note = _label(
        f"圆点为实际选择：1 表示{action_zero}，0 表示{action_one}。当前动作和奖励只影响下一次预测。",
        f"Observed points: 1 = {action_zero}, 0 = {action_one}. The current event affects the next prediction.",
    )
    figure.text(0.5, 0.015, note, ha="center", fontsize=9, color="#475569")
    figure.tight_layout(rect=(0, 0.065, 1, 0.9))
    _save(figure, output_dir, "predictions")


def plot_block_example(record: dict, output_dir: str | Path) -> None:
    """在同一条试次轴上，对照真实反馈、GRU 状态和模型预测。

    第 t 次选择前的状态、预测与第 t 次行为对齐。当前动作和奖励用于更新
    下一次选择前的状态，不能把当前反馈提前用于预测当前选择。
    """
    trials = np.asarray(record["trial_numbers"])
    actions = np.asarray(record["actions"])
    rewards = np.asarray(record["rewards"])
    states = np.asarray(record["models"]["GRU"]["states"])
    action_zero, action_one = _action_labels(record)

    figure, axes = plt.subplots(
        4, 1,
        figsize=(11.5, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1.5, 1.5]},
    )
    choice_axis, reward_axis, state_axis, prediction_axis = axes

    choice_axis.step(trials, actions, where="mid", color="#64748b", alpha=0.4)
    choice_axis.scatter(trials, actions, s=25, color="#334155", zorder=3)
    choice_axis.set_yticks([0, 1], [action_zero, action_one])
    choice_axis.set_ylim(-0.2, 1.2)
    choice_axis.set_ylabel(_label("实际选择", "Observed choice"))
    choice_axis.set_title(_label("猴子的选择与反馈", "Observed choices and outcomes"), loc="left")

    reward_axis.vlines(trials, 0, rewards, color="#c97722", linewidth=1.4, alpha=0.6)
    reward_axis.scatter(trials, rewards, s=25, color="#c97722", zorder=3)
    reward_axis.set_yticks([0, 1], [_label("无奖励", "No reward"), _label("有奖励", "Reward")])
    reward_axis.set_ylim(-0.2, 1.2)
    reward_axis.set_ylabel(_label("实际奖励", "Observed reward"))

    state_axis.plot(trials, states[:, 0], color="#2864ba", linewidth=1.8, label="h1")
    state_axis.plot(trials, states[:, 1], color="#8b5fbf", linewidth=1.8, label="h2")
    state_axis.set_ylabel(_label("隐藏状态", "Hidden state"))
    state_axis.set_title(_label("GRU：本次选择前的状态", "GRU: state before the current choice"), loc="left")
    state_axis.legend(loc="best", ncols=2, fontsize=9)

    # GRU 是此图要解释的模型；RW 作为对照，用较细的虚线区分。
    for model_name in ("GRU", "RW"):
        if model_name not in record["models"]:
            continue
        probabilities = np.asarray(record["models"][model_name]["probabilities"])
        prediction_axis.plot(
            trials,
            probabilities[:, 0],
            color=MODEL_COLORS[model_name],
            linewidth=2 if model_name == "GRU" else 1.3,
            linestyle="-" if model_name == "GRU" else "--",
            label=model_name,
        )
    prediction_axis.axhline(0.5, color="#64748b", linestyle=":", linewidth=1)
    prediction_axis.set_ylim(-0.05, 1.05)
    prediction_axis.set_yticks([0, 0.5, 1])
    prediction_axis.set_ylabel(_label(f"选择{action_zero}的概率", f"P(choose {action_zero})"))
    prediction_axis.set_title(_label("模型：本次选择前的预测", "Models: prediction before the current choice"), loc="left")
    prediction_axis.set_xlabel(_label("block 内原始试次编号", "Original trial within block"))
    prediction_axis.legend(loc="best", ncols=2, fontsize=9)

    for axis in axes:
        _trial_axis(axis, record, show_reversal_label=axis is choice_axis)
    if choice_axis.get_legend_handles_labels()[0]:
        # 图例放在坐标区上方，避免遮住 block 末段的真实选择。
        choice_axis.legend(loc="lower right", bbox_to_anchor=(1, 1.02), fontsize=9)

    figure.suptitle(
        _label("单个 block 的运行示例", "One-block example") + "\n" + _block_title(record),
        fontsize=13,
    )
    note = _label(
        "四个面板共用原始 trial 编号；红色虚线标出程序反转。隐藏状态来自模型，并非脑活动记录。\n"
        "trial t 的状态与概率均在本次选择前计算；本次动作和奖励更新的是 trial t+1 的状态与预测。",
        "Panels share original trial numbers; red dashed lines mark programmed reversal. Hidden states are model quantities.\n"
        "States and probabilities at trial t precede the choice; its action and reward update the state and prediction for trial t+1.",
    )
    figure.text(0.5, 0.018, note, ha="center", va="bottom", fontsize=9, color="#475569", linespacing=1.6)
    figure.tight_layout(rect=(0, 0.08, 1, 0.98), h_pad=1.5)
    _save(figure, output_dir, "block_example")


def plot_state_trajectories(record: dict, output_dir: str | Path) -> None:
    """上排画 T 个事件前状态，下排画含末次更新的 T+1 个二维状态。"""
    trials = np.asarray(record["trial_numbers"])
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    for model_index, model_name in enumerate(("RW", "GRU")):
        model_record = record["models"][model_name]
        states = np.asarray(model_record["states"])
        final_state = np.asarray(model_record["final_state"])
        complete_path = np.vstack((states, final_state[None]))
        coordinate_labels = _state_labels(model_name, record)
        time_axis = axes[0, model_index]
        path_axis = axes[1, model_index]

        for coordinate_index in (0, 1):
            time_axis.plot(
                trials,
                states[:, coordinate_index],
                linewidth=1.8,
                label=coordinate_labels[coordinate_index],
            )
        _trial_axis(time_axis, record, show_reversal_label=True)
        time_axis.set_title(_label(f"{model_name}：选择前状态", f"{model_name}: state before choice"))
        time_axis.set_xlabel(_label("block 内原始试次编号", "Original trial within block"))
        time_axis.set_ylabel(_label("状态值", "State value"))
        time_axis.legend(loc="best", fontsize=9)

        path_axis.plot(
            complete_path[:, 0],
            complete_path[:, 1],
            color=MODEL_COLORS[model_name],
            linewidth=1.2,
            alpha=0.65,
        )
        path_axis.scatter(
            complete_path[:, 0],
            complete_path[:, 1],
            s=14,
            color=MODEL_COLORS[model_name],
            alpha=0.4,
        )
        path_axis.scatter(
            complete_path[0, 0],
            complete_path[0, 1],
            s=65,
            color="#3c8760",
            marker="o",
            label=_label("初始状态", "Initial state"),
            zorder=3,
        )
        path_axis.scatter(
            complete_path[-1, 0],
            complete_path[-1, 1],
            s=100,
            color="#885b97",
            marker="*",
            label=_label("末次更新后", "After final update"),
            zorder=3,
        )
        path_axis.set_title(_label(f"{model_name}：完整二维轨迹", f"{model_name}: complete trajectory"))
        path_axis.set_xlabel(coordinate_labels[0])
        path_axis.set_ylabel(coordinate_labels[1])
        path_axis.legend(loc="best", fontsize=9)

    figure.suptitle(_block_title(record), fontsize=13)
    note = _label(
        "上排为每次选择之前的状态；下排还包含最后一次事件之后的状态。GRU 的 h1、h2 是隐藏坐标。",
        "Top: states before choices. Bottom: also includes the final update. GRU h1 and h2 are hidden coordinates.",
    )
    figure.text(0.5, 0.015, note, ha="center", fontsize=9, color="#475569")
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(figure, output_dir, "states")


def _decision_boundary_segment(
    readout_vector: np.ndarray,
    readout_bias: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> np.ndarray:
    """求 w·state+b=0 与可视矩形的交点，不把边界误写成经过原点。

    直接求四条边上的交点，也适用于水平或竖直的决策边界。
    如果读出权重为零，或边界没有穿过当前区域，就不画这条线。
    """
    vector_length = np.linalg.norm(readout_vector)
    if vector_length == 0:
        return np.empty((0, 2))
    normal = readout_vector / vector_length
    offset = readout_bias / vector_length
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    tolerance = 1e-10 * max(x_max - x_min, y_max - y_min, 1.0)
    candidates = []

    if abs(normal[1]) > 1e-12:
        for x_value in x_limits:
            y_value = -(normal[0] * x_value + offset) / normal[1]
            if y_min - tolerance <= y_value <= y_max + tolerance:
                candidates.append([x_value, float(np.clip(y_value, y_min, y_max))])
    if abs(normal[0]) > 1e-12:
        for y_value in y_limits:
            x_value = -(normal[1] * y_value + offset) / normal[0]
            if x_min - tolerance <= x_value <= x_max + tolerance:
                candidates.append([float(np.clip(x_value, x_min, x_max)), y_value])

    # 经过矩形角点时，同一个交点可能被两条边同时找到。
    unique_points = []
    for candidate in candidates:
        already_found = any(
            np.linalg.norm(np.asarray(candidate) - point) <= tolerance
            for point in unique_points
        )
        if not already_found:
            unique_points.append(np.asarray(candidate))
    if len(unique_points) < 2:
        return np.empty((0, 2))
    return np.asarray(unique_points[:2])


def _plot_readout(
    axis,
    field: dict,
    action_zero: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    """画偏好动作 0 的方向和等概率边界，方向箭头长度只用于展示。"""
    readout_vector = np.asarray(field["readout_vector"], dtype=float)
    readout_bias = float(field["readout_bias"])
    boundary = _decision_boundary_segment(readout_vector, readout_bias, x_limits, y_limits)
    if len(boundary) == 2:
        axis.plot(
            boundary[:, 0],
            boundary[:, 1],
            color="#f28b19",
            linestyle="--",
            linewidth=1.8,
            zorder=5,
        )

    vector_length = np.linalg.norm(readout_vector)
    if vector_length == 0:
        if readout_bias == 0:
            readout_label = _label("所有状态：选择概率各为 50%", "All states: equal choice probabilities")
        else:
            readout_label = _label("选择偏好不随状态改变", "Choice preference does not depend on state")
        axis.text(
            0.04, 0.96, readout_label,
            transform=axis.transAxes, va="top", fontsize=8, zorder=6,
        )
        return

    # w = 两个 logit 的权重之差，沿 w 移动会增加 logit_0 - logit_1。
    # 将方向归一化后画一支固定展示长度的箭头，不能把它当作状态更新量。
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    display_vector = readout_vector / vector_length * 0.17 * min(x_span, y_span)
    center = np.array([x_limits[0] + 0.15 * x_span, y_limits[0] + 0.81 * y_span])
    start = center - display_vector / 2
    end = center + display_vector / 2
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": "#f28b19", "lw": 2.6, "mutation_scale": 16},
        zorder=6,
    )
    axis.text(
        0.04,
        0.96,
        _label(f"更偏向{action_zero}", f"Prefer {action_zero}"),
        transform=axis.transAxes,
        va="top",
        fontsize=8,
        color="#8b4400",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
        zorder=6,
    )


def plot_vector_field(field: dict, model_name: str, output_dir: str | Path) -> None:
    """固定一套模型权重，汇总多个 block 来展示四种动作/奖励事件。

    颜色和细等值线表示每 trial 的欧氏更新幅度；同一模型共用色标。
    黑箭头取自各 block 的真实行为序列，并保留一步长度。箭头可以抽样，
    但坐标范围始终依据全部访问状态确定，不随抽样结果变化。
    橙色箭头表示读出方向，橙色虚线表示两个动作概率相等的状态。
    """
    grid_x = np.asarray(field["x"])
    grid_y = np.asarray(field["y"])
    observed_states = np.asarray(field["observed_states"])
    source_blocks = field["source_blocks"]
    animal_names = sorted({block["animal_name"] for block in source_blocks})
    session_ids = {(block["animal_name"], block["session_name"]) for block in source_blocks}
    block_types = {block["block_type"] for block in source_blocks}

    # 混合 what / where 时，动作 0 分别对应图像 0 / 位置 0，不能只写其中一种。
    if len(block_types) == 1:
        action_labels = _action_labels(source_blocks[0])
    else:
        action_labels = (_label("选项 0", "option 0"), _label("选项 1", "option 1"))
    if model_name == "RW":
        coordinate_labels = (f"Q({action_labels[0]})", f"Q({action_labels[1]})")
    else:
        coordinate_labels = ("h1", "h2")
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 11), sharex=True, sharey=True)

    # 上排无奖励，下排有奖励；每一列固定一个动作，与原文排列一致。
    conditions = sorted(
        field["conditions"],
        key=lambda condition: (condition["reward"], condition["action"]),
    )
    speed_max = float(field["speed_max"])
    # 完全不更新时，仍可画出零值背景；色标只显示实际存在的数值 0。
    color_max = speed_max if speed_max > 0 else 1.0
    color_levels = np.linspace(0, color_max, 21)
    contour_levels = color_levels[1:-1]

    # observed_states 包含所有 block 的状态和末次更新后的终点。
    # 因此不用抽样后的箭头来定范围，也不把不同 block 连成一条轨迹。
    x_min = min(grid_x.min(), observed_states[:, 0].min())
    x_max = max(grid_x.max(), observed_states[:, 0].max())
    y_min = min(grid_y.min(), observed_states[:, 1].min())
    y_max = max(grid_y.max(), observed_states[:, 1].max())
    arrow_data = []
    for condition in conditions:
        arrow_starts = np.asarray(condition["observed_states"]).reshape(-1, 2)
        arrow_changes = np.asarray(condition["observed_changes"]).reshape(-1, 2)
        arrow_data.append((arrow_starts, arrow_changes))
    x_margin = 0.02 * max(x_max - x_min, 0.1)
    y_margin = 0.02 * max(y_max - y_min, 0.1)
    x_limits = (x_min - x_margin, x_max + x_margin)
    y_limits = (y_min - y_margin, y_max + y_margin)

    for axis, condition, (arrow_starts, arrow_changes) in zip(axes.flat, conditions, arrow_data):
        axis.grid(False)
        speed = np.asarray(condition["speed"])
        background = axis.contourf(
            grid_x,
            grid_y,
            speed,
            levels=color_levels,
            cmap="viridis",
            alpha=0.22,
        )
        # 等值线连接“更新幅度相同”的状态，不是模型随时间移动的轨迹或流线。
        visible_levels = contour_levels[
            (contour_levels > speed.min()) & (contour_levels < speed.max())
        ]
        if len(visible_levels) > 0:
            axis.contour(
                grid_x, grid_y, speed,
                levels=visible_levels, colors="black", linewidths=0.45, alpha=0.45,
            )
        if len(arrow_starts) > 0:
            # scale=1 表示坐标上的真实一步变化；不同反馈的更新不会串成同一条线。
            axis.quiver(
                arrow_starts[:, 0],
                arrow_starts[:, 1],
                arrow_changes[:, 0],
                arrow_changes[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color="black",
                width=0.004,
                alpha=0.85,
                zorder=4,
            )
        else:
            axis.text(
                0.5, 0.05,
                _label("这些 block 中没有此事件", "No such event in the selected blocks"),
                transform=axis.transAxes, ha="center", fontsize=8,
            )
        _plot_readout(axis, field, action_labels[0], x_limits, y_limits)
        action_label = action_labels[condition["action"]]
        reward_label = _label("有奖励", "reward") if condition["reward"] else _label("无奖励", "no reward")
        arrow_count = f"{condition['n_displayed']} / {condition['n_observed']}"
        axis.set_title(_label(
            f"选择{action_label} · {reward_label}\n显示箭头：{arrow_count} 次事件",
            f"Choose {action_label} · {reward_label}\nDisplayed arrows: {arrow_count} events",
        ))
        axis.set_xlabel(coordinate_labels[0])
        axis.set_ylabel(coordinate_labels[1])
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # 单独预留色标和说明的位置，避免挤压四个面板。
    figure.subplots_adjust(left=0.08, right=0.84, bottom=0.24, top=0.86, wspace=0.22, hspace=0.34)
    colorbar_axis = figure.add_axes((0.88, 0.28, 0.025, 0.53))
    colorbar = figure.colorbar(background, cax=colorbar_axis)
    colorbar.set_label(_label("每 trial 的状态更新幅度", "State change per trial"))
    if speed_max == 0:
        colorbar.set_ticks([0])
    else:
        colorbar.locator = MaxNLocator(nbins=6)
        colorbar.update_ticks()
    source_description = _label(
        f"猴子 {', '.join(animal_names)} · {len(source_blocks)} 个 block · {len(session_ids)} 个 session · {field['n_trials']} 次反馈",
        f"Monkey {', '.join(animal_names)} · {len(source_blocks)} blocks · {len(session_ids)} sessions · {field['n_trials']} events",
    )
    figure.suptitle(
        _label(f"{model_name} · 固定模型的状态更新规则", f"{model_name} · State updates of one fixed model")
        + "\n" + source_description,
        fontsize=13,
    )
    if field["max_arrows_per_condition"] is None:
        sampling_note = _label("显示全部事件", "all events shown")
    else:
        sampling_note = _label(
            f"每类最多抽取 {field['max_arrows_per_condition']} 次事件，固定种子 {field['sample_seed']}",
            f"up to {field['max_arrows_per_condition']} events per condition, seed {field['sample_seed']}",
        )
    model_note = _label(
        "GRU 使用普通线性读出，h1、h2 不能直接当作 Q 值。",
        "GRU has a general linear readout; h1 and h2 are not Q values.",
    ) if model_name == "GRU" else _label(
        "RW 的两个坐标是两个选项的价值 Q。",
        "The two RW coordinates are the option values Q.",
    )
    note = _label(
        "颜色与细等值线：假设状态下每 trial 的更新幅度（紫色小，黄色大）；等值线不是轨迹。\n"
        f"黑箭头：真实行为驱动的模型更新；{sampling_note}。\n"
        f"橙色箭头：更偏向{action_labels[0]}的方向，长度仅作展示；橙色虚线：两个动作概率各为 50%。\n"
        "what 按图像、where 按位置编码；各 block 独立初始化，共享同一套权重。\n"
        f"{model_note} 不同模型的色标独立，不能据此比较学习率。",
        "Colors and contours: update magnitude at hypothetical states (purple: small, yellow: large), not trajectories.\n"
        f"Black arrows: behavior-driven model updates; {sampling_note}.\n"
        f"Orange arrow: prefer {action_labels[0]} (display length only); dashed line: equal choice probabilities.\n"
        "what uses image codes; where uses location codes. Blocks reset independently and share one set of weights.\n"
        f"{model_note} Color scales are model-specific and do not compare learning rates.",
    )
    figure.text(0.5, 0.035, note, ha="center", va="bottom", fontsize=9, color="#475569", linespacing=1.65)
    _save(figure, output_dir, f"dynamics_{model_name.lower()}")


def plot_training_history(history: dict, output_dir: str | Path) -> None:
    """如实标出训练损失和验证损失使用的权重时点，并标记最佳轮次。"""
    epochs = np.asarray(history["epoch"])
    train_nll = np.asarray(history["train_nll"])
    validation_nll = np.asarray(history["validation_nll"])
    best_epoch = history["best_epoch"]
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.plot(
        epochs,
        train_nll,
        color="#2864ba",
        linewidth=1.8,
        label=_label("训练 NLL（更新前）", "Training NLL (before update)"),
    )
    axis.plot(
        epochs,
        validation_nll,
        color="#c97722",
        linewidth=1.8,
        label=_label("验证 NLL（更新后）", "Validation NLL (after update)"),
    )
    if epochs[0] <= best_epoch <= epochs[-1]:
        axis.axvline(
            best_epoch,
            color="#885b97",
            linestyle="--",
            linewidth=1.2,
            label=_label(f"最佳轮次：{best_epoch}", f"Best epoch: {best_epoch}"),
        )
    axis.set_xlabel(_label("训练轮次", "Epoch"))
    axis.set_ylabel(_label("每次试次的平均 NLL", "Mean NLL per trial"))
    axis.set_title(_label("GRU 参数拟合过程", "GRU parameter fitting"))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
    axis.legend(loc="best", fontsize=9)
    note = _label(
        "每轮先计算训练损失，再更新权重；验证损失使用更新后的权重。虚线标出最终采用的最佳轮次。",
        "Training loss precedes the update; validation loss follows it. The dashed line marks the selected epoch.",
    )
    figure.text(0.5, 0.015, note, ha="center", fontsize=9, color="#475569")
    figure.tight_layout(rect=(0, 0.075, 1, 1))
    _save(figure, output_dir, "training")


def _publication_style():
    """主图使用统一的论文排版；作用域结束后恢复其余图的样式。"""
    return plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.7,
            "axes.facecolor": "white",
            "axes.grid": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "mathtext.fontset": "dejavusans",
        }
    )


def plot_dynamics_comparison(fields: dict, output_dir: str | Path) -> None:
    """汇总测试集，在同一张图中比较 RW 与 GRU 的四种条件动力学。

    每一行对应一套已经拟合好的模型，每一列固定一种动作和奖励组合。
    背景在状态网格上计算；黑箭头使用传入的全部真实事件，不在绘图时抽样。
    同一模型共享坐标和色标，但 Q 与 h 的单位不同，因此两行独立标色。
    数据范围、读出边界和图例含义在独立图注中解释，不写成长段图内文字。
    """
    model_names = ("RW", "GRU")
    event_order = ((0, 0), (0, 1), (1, 0), (1, 1))
    # 直接调浅颜色，避免半透明网格的小三角形重叠形成额外纹理。
    base_colors = matplotlib.colormaps["viridis"](np.linspace(0, 1, 256))[:, :3]
    pale_colors = 0.73 + 0.27 * base_colors
    pale_colormap = matplotlib.colors.ListedColormap(pale_colors)

    with _publication_style():
        figure = plt.figure(figsize=(8.0, 4.2))
        layout = figure.add_gridspec(
            2, 5,
            width_ratios=[1, 1, 1, 1, 0.07],
            left=0.10,
            right=0.90,
            bottom=0.12,
            top=0.91,
            wspace=0.30,
            hspace=0.40,
        )

        for row_index, model_name in enumerate(model_names):
            field = fields[model_name]
            grid_x = np.asarray(field["x"])
            grid_y = np.asarray(field["y"])
            all_states = np.asarray(field["observed_states"])
            conditions = {
                (condition["action"], condition["reward"]): condition
                for condition in field["conditions"]
            }

            # 全部访问状态包括每个 block 的末状态。坐标范围不由箭头抽样决定。
            x_min = min(float(grid_x.min()), float(all_states[:, 0].min()))
            x_max = max(float(grid_x.max()), float(all_states[:, 0].max()))
            y_min = min(float(grid_y.min()), float(all_states[:, 1].min()))
            y_max = max(float(grid_y.max()), float(all_states[:, 1].max()))
            x_margin = 0.02 * max(x_max - x_min, 0.1)
            y_margin = 0.02 * max(y_max - y_min, 0.1)
            x_limits = (x_min - x_margin, x_max + x_margin)
            y_limits = (y_min - y_margin, y_max + y_margin)

            speed_max = float(field["speed_max"])
            color_max = speed_max if speed_max > 0 else 1.0
            # 四个条件统一使用 20 个等值层级，便于看清较小幅度区域的结构。
            contour_levels = np.linspace(0, color_max, 22)[1:-1]
            boundary = _decision_boundary_segment(
                np.asarray(field["readout_vector"]),
                float(field["readout_bias"]),
                x_limits,
                y_limits,
            )
            row_axes = []

            for column_index, event in enumerate(event_order):
                axis = figure.add_subplot(layout[row_index, column_index])
                row_axes.append(axis)
                condition = conditions[event]
                speed = np.asarray(condition["speed"])
                # 平滑填色，避免大量色阶边缘被误读成额外的等值线。
                background = axis.pcolormesh(
                    grid_x,
                    grid_y,
                    speed,
                    shading="gouraud",
                    cmap=pale_colormap,
                    vmin=0,
                    vmax=color_max,
                    rasterized=True,
                )
                # 细线连接更新幅度相同的起点；它们不是连续的状态轨迹。
                visible_levels = contour_levels[
                    (contour_levels > speed.min()) & (contour_levels < speed.max())
                ]
                if len(visible_levels) > 0:
                    axis.contour(
                        grid_x,
                        grid_y,
                        speed,
                        levels=visible_levels,
                        colors="#526064",
                        linewidths=0.35,
                        alpha=0.45,
                    )

                starts = np.asarray(condition["observed_states"]).reshape(-1, 2)
                changes = np.asarray(condition["observed_changes"]).reshape(-1, 2)
                if len(starts) > 0:
                    # 保留实际一步位移，不归一化或放大，也不连接不同 block。
                    axis.quiver(
                        starts[:, 0],
                        starts[:, 1],
                        changes[:, 0],
                        changes[:, 1],
                        angles="xy",
                        scale_units="xy",
                        scale=1,
                        color="black",
                        width=0.0040,
                        headwidth=4.0,
                        headlength=5.0,
                        headaxislength=4.5,
                        minlength=0,
                        alpha=0.58,
                        zorder=3,
                    )
                else:
                    axis.text(
                        0.5, 0.07, r"$n=0$",
                        transform=axis.transAxes,
                        ha="center",
                        fontsize=8,
                    )

                # 只保留等概率边界；省去额外的读出方向箭头，避免混淆两种箭头。
                if len(boundary) == 2:
                    axis.plot(
                        boundary[:, 0],
                        boundary[:, 1],
                        color="#d97706",
                        linestyle=(0, (4, 3)),
                        linewidth=0.9,
                        zorder=4,
                    )

                axis.set_xlim(x_limits)
                axis.set_ylim(y_limits)
                axis.set_aspect("equal", adjustable="box")
                axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
                axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
                axis.set_xlabel(r"$Q_0$" if model_name == "RW" else r"$h_1$", labelpad=2)
                if column_index == 0:
                    axis.set_ylabel(r"$Q_1$" if model_name == "RW" else r"$h_2$", labelpad=2)
                else:
                    axis.tick_params(axis="y", labelleft=False)
                if row_index == 0:
                    action, reward = event
                    axis.set_title(rf"$(a_t,r_t)=({action},{reward})$", pad=13)
                panel_letter = chr(ord("a") + row_index * len(event_order) + column_index)
                axis.text(
                    0.02, 1.035, panel_letter,
                    transform=axis.transAxes,
                    fontweight="bold",
                    fontsize=9,
                    va="bottom",
                )

            # 每行各用一个色标；不同模型的颜色不能用来比较学习率。
            colorbar_axis = figure.add_subplot(layout[row_index, 4])
            colorbar = figure.colorbar(background, cax=colorbar_axis)
            colorbar.set_label(
                r"$\|\Delta Q\|_2$" if model_name == "RW" else r"$\|\Delta h\|_2$",
                labelpad=5,
            )
            if speed_max == 0:
                colorbar.set_ticks([0])
            else:
                colorbar.locator = MaxNLocator(nbins=3)
                colorbar.update_ticks()
            colorbar.outline.set_linewidth(0.5)

            # 行名只出现一次；其余解释由外部图注承担。
            first_position = row_axes[0].get_position()
            colorbar_position = colorbar_axis.get_position()
            colorbar_axis.set_position(
                [
                    colorbar_position.x0,
                    first_position.y0,
                    colorbar_position.width,
                    first_position.height,
                ]
            )
            figure.text(
                0.015,
                (first_position.y0 + first_position.y1) / 2,
                model_name,
                rotation=90,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        _save(figure, output_dir, "dynamics_comparison")


def plot_model_comparison(comparison: dict, output_dir: str | Path) -> None:
    """比较同一测试集的预测；样本信息和结论范围放在独立图注中。"""
    model_names = ["RW", "GRU"]
    nll_values = [comparison["models"][name]["nll"] for name in model_names]
    accuracy_values = [
        100 * comparison["models"][name]["accuracy"] for name in model_names
    ]
    colors = [MODEL_COLORS[name] for name in model_names]

    with _publication_style():
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
        figure.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.90, wspace=0.45)
        for panel, axis, values in zip("ab", axes, (nll_values, accuracy_values)):
            bars = axis.bar(model_names, values, width=0.55, color=colors)
            axis.margins(x=0.26)
            axis.text(
                -0.20, 1.04, panel,
                transform=axis.transAxes,
                fontsize=10,
                fontweight="bold",
                va="bottom",
            )
            for bar, value in zip(bars, values):
                number = f"{value:.3f}" if axis is axes[0] else f"{value:.1f}"
                axis.annotate(
                    number,
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # 柱状图保留零基线，不借截断纵轴放大模型之间的差异。
        axes[0].set_ylim(0, max(max(nll_values) * 1.22, 0.1))
        axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
        axes[0].set_ylabel("Test NLL")
        axes[1].set_ylim(0, 100)
        axes[1].set_yticks(np.arange(0, 101, 20))
        axes[1].set_ylabel("Accuracy (%)")
        _save(figure, output_dir, "model_comparison")


def plot_one_step(step: dict, output_dir: str | Path) -> None:
    """用两个紧凑状态平面展示同一输入引起的一次更新。

    箭头使用计算得到的真实端点，既不归一化，也不放大位移。
    每个平面使用等比例坐标；不同模型的坐标单位不相同。
    状态来源、试次信息和完整解释保存在独立图注中。
    """
    with _publication_style():
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
        figure.subplots_adjust(left=0.09, right=0.985, bottom=0.25, top=0.85, wspace=0.45)

        for panel, axis, model_name in zip("ab", axes, ("RW", "GRU")):
            result = step["models"][model_name]
            before = np.asarray(result["before"], dtype=float)
            after = np.asarray(result["after"], dtype=float)
            change = after - before
            color = MODEL_COLORS[model_name]
            axis.set_title(model_name, loc="left", pad=6)
            axis.text(
                -0.23, 1.03, panel,
                transform=axis.transAxes,
                fontsize=10,
                fontweight="bold",
                va="bottom",
            )
            if model_name == "RW":
                axis.set_xlabel(r"$Q_0$")
                axis.set_ylabel(r"$Q_1$")
            else:
                axis.set_xlabel(r"$h_1$")
                axis.set_ylabel(r"$h_2$")

            # 两个坐标方向共用跨度，确保箭头的角度不受坐标拉伸影响。
            endpoints = np.vstack([before, after])
            center = endpoints.mean(axis=0)
            coordinate_span = max(float(np.ptp(endpoints, axis=0).max()), 0.1)
            half_width = coordinate_span * 0.75
            axis.set_xlim(center[0] - half_width, center[0] + half_width)
            axis.set_ylim(center[1] - half_width, center[1] + half_width)
            axis.set_aspect("equal", adjustable="box")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))

            unchanged = np.array_equal(before, after)
            if not unchanged:
                axis.quiver(
                    before[0], before[1], change[0], change[1],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color=color,
                    width=0.006,
                    headwidth=5,
                    headlength=6,
                    minlength=0,
                    zorder=3,
                )
            axis.scatter(
                before[0], before[1],
                s=38,
                facecolors="white",
                edgecolors=color,
                linewidths=1.2,
                zorder=4,
            )
            axis.scatter(after[0], after[1], s=16, color=color, zorder=5)
            if unchanged:
                # 重合的端点使用同心标记，不为展示效果人为制造迁移。
                axis.annotate(
                    r"$\Delta s=0$",
                    xy=before,
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

            before_probability = result["probabilities_before"][0]
            after_probability = result["probabilities_after"][0]
            axis.text(
                0.5, -0.32,
                rf"$P(a=0):\ {before_probability:.3f}\rightarrow {after_probability:.3f}$",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=8,
            )

        # 共享输入和标记图例只出现一次；假设输入必须明确区别于真实事件。
        input_label = rf"$(a_t,r_t)=({step['action']},{step['reward']})$"
        if not step["is_observed_transition"]:
            input_label += "  (Hypothetical)"
        figure.text(0.5, 0.97, input_label, ha="center", va="top", fontsize=9)
        legend_handles = [
            plt.Line2D(
                [], [], linestyle="none", marker="o", markersize=5,
                markerfacecolor="white", markeredgecolor="#333333",
                markeredgewidth=1.0, label=r"$s_t$",
            ),
            plt.Line2D(
                [], [], linestyle="none", marker="o", markersize=4,
                markerfacecolor="#333333", markeredgecolor="#333333",
                label=r"$s_{t+1}$",
            ),
        ]
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            handletextpad=0.35,
            columnspacing=1.3,
            borderaxespad=0,
        )
        _save(figure, output_dir, "one_step")

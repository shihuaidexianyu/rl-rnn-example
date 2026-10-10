"""把一个 block 的行为、预测和状态更新画成便于阅读的教学图。

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


def plot_vector_field(field: dict, model_name: str, block: dict, output_dir: str | Path) -> None:
    """用论文 Fig. 4a 的读图方式展示四种动作/奖励事件。

    颜色和细等值线表示每 trial 的欧氏更新幅度；同一模型共用色标。
    黑箭头默认只画真实行为序列中该事件对应的模型更新，并保留一步长度。
    橙色箭头表示读出方向，橙色虚线表示两个动作概率相等的状态。
    """
    grid_x = np.asarray(field["x"])
    grid_y = np.asarray(field["y"])
    observed_states = np.asarray(field["observed_states"])
    has_observed_events = field["has_observed_events"]
    coordinate_labels = _state_labels(model_name, block)
    action_labels = _action_labels(block)
    figure, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)

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

    # 四幅图使用相同的坐标域，包含背景网格和实际绘制的箭头终点。
    x_min = min(grid_x.min(), observed_states[:, 0].min())
    x_max = max(grid_x.max(), observed_states[:, 0].max())
    y_min = min(grid_y.min(), observed_states[:, 1].min())
    y_max = max(grid_y.max(), observed_states[:, 1].max())
    arrow_data = []
    for condition in conditions:
        if has_observed_events:
            arrow_starts = np.asarray(condition["observed_states"]).reshape(-1, 2)
            arrow_changes = np.asarray(condition["observed_changes"]).reshape(-1, 2)
        else:
            # 未提供行为事件时，仅以稀疏网格演示更新规则；图注会说明是假设状态。
            stride = max(1, int(np.ceil(max(grid_x.shape) / 7)))
            arrow_starts = np.column_stack(
                (grid_x[::stride, ::stride].ravel(), grid_y[::stride, ::stride].ravel())
            )
            arrow_changes = np.column_stack(
                (
                    np.asarray(condition["dx"])[::stride, ::stride].ravel(),
                    np.asarray(condition["dy"])[::stride, ::stride].ravel(),
                )
            )
        arrow_data.append((arrow_starts, arrow_changes))
        if len(arrow_starts) == 0:
            continue
        endpoints = arrow_starts + arrow_changes
        endpoint_x = endpoints[:, 0]
        endpoint_y = endpoints[:, 1]
        x_min = min(x_min, endpoint_x.min())
        x_max = max(x_max, endpoint_x.max())
        y_min = min(y_min, endpoint_y.min())
        y_max = max(y_max, endpoint_y.max())
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
        elif has_observed_events:
            axis.text(
                0.5, 0.05,
                _label("该 block 中没有此事件", "No such event in this block"),
                transform=axis.transAxes, ha="center", fontsize=8,
            )
        _plot_readout(axis, field, action_labels[0], x_limits, y_limits)
        action_label = action_labels[condition["action"]]
        reward_label = _label("有奖励", "reward") if condition["reward"] else _label("无奖励", "no reward")
        axis.set_title(_label(f"选择{action_label} · {reward_label}", f"Choose {action_label} · {reward_label}"))
        axis.set_xlabel(coordinate_labels[0])
        axis.set_ylabel(coordinate_labels[1])
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # 单独预留色标和说明的位置，避免挤压四个面板。
    figure.subplots_adjust(left=0.08, right=0.84, bottom=0.21, top=0.9, wspace=0.22, hspace=0.27)
    colorbar_axis = figure.add_axes((0.88, 0.25, 0.025, 0.57))
    colorbar = figure.colorbar(background, cax=colorbar_axis)
    colorbar.set_label(_label("每 trial 的状态更新幅度", "State change per trial"))
    if speed_max == 0:
        colorbar.set_ticks([0])
    else:
        colorbar.locator = MaxNLocator(nbins=6)
        colorbar.update_ticks()
    figure.suptitle(f"{model_name} · {_block_title(block)}", fontsize=13)
    if has_observed_events:
        arrow_note = _label(
            "黑箭头：该 block 的真实选择与反馈所驱动的模型更新，按四种事件分别显示。",
            "Black arrows: model updates driven by observed choices and outcomes in this block, grouped by event.",
        )
    else:
        arrow_note = _label(
            "黑箭头：假设网格状态的一步更新；未提供行为事件，不代表实际访问过这些状态。",
            "Black arrows: updates at hypothetical grid states; observed behavioral events were not supplied.",
        )
    note = _label(
        "颜色与细等值线：每 trial 的更新幅度（紫色小，黄色大）；等值线不表示轨迹。\n"
        f"{arrow_note}\n"
        f"橙色箭头：更偏向{action_labels[0]}的方向，长度仅作展示；橙色虚线：两个动作概率各为 50%。\n"
        "GRU 使用普通线性读出，h1、h2 不能直接当作 Q 值；不同模型的色标独立，不能据此比较学习率。",
        "Colors and thin contours: state change per trial (purple: small, yellow: large); contours are not trajectories.\n"
        f"{arrow_note}\n"
        f"Orange arrow: increasing preference for {action_labels[0]} (display length only); dashed line: equal choice probabilities.\n"
        "GRU uses a general linear readout; h1 and h2 are not Q values. Model-specific color scales do not compare learning rates.",
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

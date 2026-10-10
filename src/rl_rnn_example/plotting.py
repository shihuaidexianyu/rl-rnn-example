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


def plot_vector_field(field: dict, model_name: str, block: dict, output_dir: str | Path) -> None:
    """画四种动作/奖励事件的真实一步更新，箭头不归一化。

    四个面板采用 field 给出的同一坐标域和相同的箭头比例。
    背景表示该状态下的 P(action=0)，固定使用 0–1 的色标。
    """
    grid_x = np.asarray(field["x"])
    grid_y = np.asarray(field["y"])
    probability = np.asarray(field["probability"])
    observed_states = np.asarray(field["observed_states"])
    coordinate_labels = _state_labels(model_name, block)
    action_labels = _action_labels(block)
    figure, axes = plt.subplots(2, 2, figsize=(11, 9.5), sharex=True, sharey=True)

    # 坐标域也覆盖真实更新的终点，不能因为起点位于网格边缘而裁掉箭头。
    # 四种事件使用同一范围，因此各面板之间仍可直接比较同一模型的更新。
    x_min = min(grid_x.min(), observed_states[:, 0].min())
    x_max = max(grid_x.max(), observed_states[:, 0].max())
    y_min = min(grid_y.min(), observed_states[:, 1].min())
    y_max = max(grid_y.max(), observed_states[:, 1].max())
    for condition in field["conditions"]:
        endpoint_x = grid_x + np.asarray(condition["dx"])
        endpoint_y = grid_y + np.asarray(condition["dy"])
        x_min = min(x_min, endpoint_x.min())
        x_max = max(x_max, endpoint_x.max())
        y_min = min(y_min, endpoint_y.min())
        y_max = max(y_max, endpoint_y.max())
    x_margin = 0.04 * (x_max - x_min)
    y_margin = 0.04 * (y_max - y_min)

    for axis, condition in zip(axes.flat, field["conditions"]):
        axis.grid(False)
        background = axis.pcolormesh(
            grid_x,
            grid_y,
            probability,
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
            shading="auto",
            alpha=0.35,
            rasterized=True,
        )
        # scale=1 表示坐标上的真实一步变化，而不是单位长度的方向箭头。
        axis.quiver(
            grid_x,
            grid_y,
            condition["dx"],
            condition["dy"],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#263445",
            width=0.003,
            alpha=0.8,
            zorder=3,
        )
        # 轨迹含全部 T+1 个状态，不能遗漏最后一次反馈之后的更新。
        axis.plot(
            observed_states[:, 0],
            observed_states[:, 1],
            color="#475569",
            alpha=0.25,
            linewidth=1.2,
            zorder=2,
        )
        axis.scatter(observed_states[0, 0], observed_states[0, 1], s=32, color="#3c8760", zorder=4)
        axis.scatter(
            observed_states[-1, 0],
            observed_states[-1, 1],
            s=40,
            color="#885b97",
            marker="x",
            zorder=4,
        )
        action_label = action_labels[condition["action"]]
        reward_label = _label("有奖励", "reward") if condition["reward"] else _label("无奖励", "no reward")
        axis.set_title(_label(f"选择{action_label} · {reward_label}", f"Choose {action_label} · {reward_label}"))
        axis.set_xlabel(coordinate_labels[0])
        axis.set_ylabel(coordinate_labels[1])
        axis.set_xlim(x_min - x_margin, x_max + x_margin)
        axis.set_ylim(y_min - y_margin, y_max + y_margin)
        axis.set_aspect("equal", adjustable="box")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # 单独预留色标和说明的位置，避免挤压四个面板。
    figure.subplots_adjust(left=0.08, right=0.84, bottom=0.14, top=0.9, wspace=0.22, hspace=0.27)
    colorbar_axis = figure.add_axes((0.88, 0.2, 0.025, 0.62))
    colorbar = figure.colorbar(background, cax=colorbar_axis)
    colorbar.set_label(_label(f"选择{action_labels[0]}的概率", f"P(choose {action_labels[0]})"))
    figure.suptitle(f"{model_name} · {_block_title(block)}", fontsize=13)
    note = _label(
        "箭头为真实的一步状态更新；浅灰线为该 block 的完整状态轨迹。\n"
        "圆圈为初始状态，叉号为末次更新后状态。RW 与 GRU 的状态单位不同，箭头长度不用于比较学习率。",
        "Arrows show one actual update; the light trajectory contains all states in this block.\n"
        "Circle: initial state. Cross: after final update. RW and GRU have different state units;\n"
        "arrow lengths do not compare learning rates.",
    )
    figure.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=9, color="#475569", linespacing=1.4)
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

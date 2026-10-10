"""一次生成章节的五张插图；不训练模型，不修改项目代码。

在项目目录执行：
    uv run python doc/figures/build_chapter_figures.py

任务图展示实验流程与 0.7/0.3 奖励概率反转，不代表某个真实 block。
两张结果图直接使用已保存的测试记录与模型。
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rl_rnn_example.analysis import (
    compute_model_dynamics,
    load_results,
    summarize_model_comparison,
)
from rl_rnn_example import plotting as project_plotting


COLORS = {"RW": "#c97722", "GRU": "#2864ba"}
INK = "#26313c"
MUTED = "#68727d"
REVERSAL = "#b74646"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
    }
)


def save_figure(figure, name):
    """文稿引用 SVG 矢量图；同时保存 300 dpi PNG 供预览。"""
    figure.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    figure.savefig(OUTPUT_DIR / f"{name}.svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def arrow(axis, start, end, color=INK, width=0.9):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=width,
        color=color,
        transform=axis.transAxes,
    )
    axis.add_patch(patch)


def node(axis, center, width, height, text, edge=INK, face="white", fontsize=9):
    """使用归一化坐标排布结构图，避免图像分辨率影响布局。"""
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.009,rounding_size=0.018",
        transform=axis.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=0.9,
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=fontsize, transform=axis.transAxes)


def plot_reversal_task():
    """任务示意：单试次流程和 block 内概率反转，不混入真实行为曲线。"""
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Circle, Polygon, Rectangle

    chinese = FontProperties(family="SimSong", weight="normal")
    ink = "#202020"
    with plt.rc_context({"mathtext.fontset": "stix", "svg.fonttype": "path"}):
        figure = plt.figure(figsize=(7.2, 4.3))
        task = figure.add_axes((0.055, 0.55, 0.91, 0.41))
        task.set(xlim=(0, 10), ylim=(0, 2.9), aspect="equal")
        task.set_axis_off()

        def label(x, y, words, size=10, ha="center"):
            task.text(x, y, words, fontproperties=chinese, fontsize=size,
                      ha=ha, va="center", color=ink)

        def connect(start, finish):
            task.add_patch(FancyArrowPatch(
                start, finish, arrowstyle="->", mutation_scale=9,
                linewidth=0.8, color=ink, shrinkA=0, shrinkB=0,
            ))

        def display(center, chosen=False):
            # 方框表示刺激呈现区域；几何形状仅代指两幅不同的图像。
            task.add_patch(Rectangle((center - 1.10, 0.88), 2.20, 1.15,
                                     edgecolor=ink, facecolor="white", linewidth=0.8))
            task.add_patch(Circle((center - 0.48, 1.46), 0.15,
                                  edgecolor=ink, facecolor="#d7d7d7", linewidth=0.7))
            task.add_patch(Polygon(
                [(center + 0.48, 1.64), (center + 0.29, 1.29), (center + 0.67, 1.29)],
                edgecolor=ink, facecolor="white", linewidth=0.8,
            ))
            if chosen:
                # 虚线圈仅是解释图中的选择标记，不表示额外的实验提示。
                task.add_patch(Circle((center - 0.48, 1.46), 0.29,
                                      edgecolor=ink, facecolor="none",
                                      linestyle=(0, (3, 2)), linewidth=0.8))

        label(0.02, 2.74, "a", size=12, ha="left")
        label(0.42, 2.74, "一次试次", size=11, ha="left")
        for x, words in ((1.47, "呈现两个选项"), (4.55, "选择其中一个"), (8.25, "所选项的反馈")):
            label(x, 2.30, words)
        display(1.47)
        display(4.55, chosen=True)
        connect((2.63, 1.46), (3.39, 1.46))
        task.plot([5.70, 6.60], [1.46, 1.46], color=ink, linewidth=0.8)
        task.plot([6.60, 6.60], [1.08, 1.85], color=ink, linewidth=0.8)
        connect((6.60, 1.85), (7.00, 1.85))
        connect((6.60, 1.08), (7.00, 1.08))
        label(7.22, 1.85, r"奖励  $r_t=1$", ha="left")
        label(7.22, 1.08, r"无奖励  $r_t=0$", ha="left")
        label(8.02, 1.46, "或", size=9)

        # 概率线是实验规则的示意；中间的跳变位置不对应任一真实 block。
        probability = figure.add_axes((0.14, 0.15, 0.80, 0.34))
        reversal = 40.5
        probability.step([1, reversal, 80], [0.7, 0.3, 0.3], where="post",
                         color=ink, linewidth=1.25, label="选项 0")
        probability.step([1, reversal, 80], [0.3, 0.7, 0.7], where="post",
                         color="#666666", linestyle=(0, (4, 3)), linewidth=1.1,
                         label="选项 1")
        probability.axvline(reversal, color="#999999", linestyle=(0, (1, 3)), linewidth=0.7)
        probability.set(xlim=(1, 80), ylim=(0, 1.04), xticks=(1, 80), yticks=(0, 0.3, 0.7, 1))
        probability.set_yticklabels(["0", "0.3", "0.7", "1"])
        probability.set_xlabel("块内试次", fontproperties=chinese, fontsize=10, labelpad=3)
        probability.set_ylabel("奖励概率", fontproperties=chinese, fontsize=10, labelpad=5)
        probability.text(0.50, 0.98, "无提示反转（位置随机）", transform=probability.transAxes,
                         fontproperties=chinese, ha="center", va="top", fontsize=9,
                         bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        probability.text(-0.062, 1.19, "b", transform=probability.transAxes,
                         fontproperties=chinese, fontsize=12, va="center")
        probability.text(-0.015, 1.19, "一个 block（80 次试次）", transform=probability.transAxes,
                         fontproperties=chinese, fontsize=11, va="center")
        probability.legend(loc="upper right", bbox_to_anchor=(1.02, 1.33),
                           ncol=2, frameon=False, prop=chinese,
                           handlelength=1.6, columnspacing=1.2)
        probability.grid(False)
        probability.spines[["right", "top"]].set_visible(False)
        probability.spines[["left", "bottom"]].set_linewidth(0.65)
        probability.tick_params(labelsize=8.5, width=0.65, length=3)
        save_figure(figure, "reversal_task")


def _structure_canvas(width, height, figsize):
    """普通黑白教材图：宋体说明、衬线数学符号、不填充色块。"""
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle

    chinese = FontProperties(family="SimSong", weight="normal")
    figure, axis = plt.subplots(figsize=figsize)
    axis.set(xlim=(0, width), ylim=(0, height), aspect="equal")
    axis.set_axis_off()
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.98)

    def label(x, y, text, size=12, ha="center"):
        axis.text(x, y, text, fontsize=size, fontproperties=chinese,
                  ha=ha, va="center", color="#202020")

    def arrow_between(start, end):
        axis.add_patch(FancyArrowPatch(
            start, end, arrowstyle="->", mutation_scale=10,
            linewidth=0.85, color="#202020", shrinkA=0, shrinkB=0,
        ))

    def operation(x, y, text, width=1.05, height=0.58):
        axis.add_patch(Rectangle(
            (x - width / 2, y - height / 2), width, height,
            edgecolor="#202020", facecolor="white", linewidth=0.85,
        ))
        label(x, y, text, size=11)

    return figure, label, arrow_between, operation


def plot_rw_architecture():
    """RW 单独成图：当前读出与反馈更新在两个方向展开。"""
    with plt.rc_context({"mathtext.fontset": "stix", "svg.fonttype": "path"}):
        figure, label, arrow_between, operation = _structure_canvas(
            7.25, 3.05, figsize=(6.4, 2.6)
        )
        label(1.10, 1.25, r"$\mathbf{Q}_t$", size=15)
        label(1.10, 0.91, r"$(Q_t(0),\,Q_t(1))$", size=10)
        label(6.10, 1.25, r"$\mathbf{Q}_{t+1}$", size=15)
        label(6.10, 0.91, r"$(Q_{t+1}(0),\,Q_{t+1}(1))$", size=10)

        # 当前选择读出仅依赖 Q_t；当前真实反馈沿下方进入更新模块。
        label(1.10, 2.73, r"$\mathbf{p}_t$", size=15)
        arrow_between((1.10, 1.51), (1.10, 2.47))
        label(1.33, 2.00, r"$\mathrm{softmax}(\beta\mathbf{Q}_t)$", size=11, ha="left")

        operation(3.65, 1.25, "价值更新", width=1.13)
        arrow_between((1.49, 1.25), (3.07, 1.25))
        arrow_between((4.23, 1.25), (5.60, 1.25))
        label(3.65, 0.25, r"$(a_t,\,r_t)$", size=13)
        arrow_between((3.65, 0.49), (3.65, 0.95))
        save_figure(figure, "rw_architecture")


def plot_gru_architecture():
    """显示试次间时序，再用简洁的主路径与弧线展开 GRU 的内部计算。"""
    from matplotlib.path import Path as DrawingPath
    from matplotlib.patches import Circle

    with plt.rc_context({"mathtext.fontset": "stix", "svg.fonttype": "path"}):
        figure, label, arrow_between, operation = _structure_canvas(
            13.2, 9.6, figsize=(7.4, 5.4)
        )
        axis = figure.axes[0]
        ink = "#202020"

        def curve(start, control_1, control_2, end):
            """三次贝塞尔曲线只用于旁路，使主要计算路径仍保持清楚。"""
            path = DrawingPath(
                [start, control_1, control_2, end],
                [DrawingPath.MOVETO, DrawingPath.CURVE4,
                 DrawingPath.CURVE4, DrawingPath.CURVE4],
            )
            axis.add_patch(FancyArrowPatch(
                path=path, arrowstyle="->", mutation_scale=10,
                linewidth=0.85, color=ink, zorder=1,
            ))

        def junction(x, y):
            axis.add_patch(Circle((x, y), 0.030, facecolor=ink, edgecolor="none", zorder=3))

        def elementwise(x, y, symbol):
            axis.add_patch(Circle((x, y), 0.20, facecolor="white",
                                  edgecolor=ink, linewidth=0.8, zorder=3))
            if symbol == r"$\odot$":
                # 外圆和中心实心点共同组成逐元素乘号，避免再套一层圆圈。
                axis.add_patch(Circle((x, y), 0.045, facecolor=ink,
                                      edgecolor="none", zorder=4))
            else:
                label(x, y, symbol, size=13)

        # a：同一个 GRU 在相邻试次重复使用；每个 h 都是二维向量。
        label(0.22, 9.25, "a", size=12, ha="left")
        label(0.70, 9.25, "试次间的状态传递", size=11, ha="left")
        state_y = 7.95
        for x, text in ((0.92, r"$\mathbf{h}_{t-1}$"),
                        (5.36, r"$\mathbf{h}_t$"),
                        (10.80, r"$\mathbf{h}_{t+1}$")):
            label(x, state_y, text, size=15)
        for x in (3.0, 8.0):
            operation(x, state_y, r"$\mathrm{GRU}_{\theta}$", width=1.24, height=0.65)
        arrow_between((1.56, state_y), (2.35, state_y))
        arrow_between((3.65, state_y), (4.94, state_y))
        arrow_between((5.79, state_y), (7.35, state_y))
        arrow_between((8.65, state_y), (10.16, state_y))
        label(3.0, 7.05, r"$\mathbf{x}_{t-1}$", size=13)
        label(8.0, 7.05, r"$\mathbf{x}_{t}=(a_t,r_t)$", size=13)
        arrow_between((3.0, 7.31), (3.0, 7.59))
        arrow_between((8.0, 7.31), (8.0, 7.59))
        label(5.36, 9.25, r"$\mathbf{p}_t$", size=15)
        arrow_between((5.36, 8.25), (5.36, 8.94))
        label(5.72, 8.71, "线性读出 → softmax", size=10, ha="left")

        # b：候选状态计算沿水平主路径展开；门控和保留路径从侧面汇入。
        label(0.22, 6.25, "b", size=12, ha="left")
        label(0.70, 6.25, "GRU 的内部计算", size=11, ha="left")
        main_y = 3.35
        label(0.80, main_y, r"$\mathbf{h}_{t}$", size=15)
        arrow_between((1.19, main_y), (1.94, main_y))
        operation(2.90, main_y, r"$W_h\mathbf{h}_t+\mathbf{b}_h$", width=1.88, height=0.60)
        arrow_between((3.88, main_y), (4.42, main_y))
        elementwise(4.65, main_y, r"$\odot$")
        arrow_between((4.88, main_y), (6.12, main_y))
        elementwise(6.35, main_y, r"$+$")
        arrow_between((6.58, main_y), (7.22, main_y))
        operation(7.80, main_y, r"$\tanh$", width=1.06, height=0.60)
        arrow_between((8.36, main_y), (9.62, main_y))
        label(8.91, 2.94, r"$\tilde{\mathbf{h}}_{t+1}$", size=13)
        elementwise(9.85, main_y, r"$\odot$")

        # 一条拱弧直接传递旧状态，避免沿图边缘作长矩形折返。
        junction(1.50, main_y)
        curve((1.50, main_y), (1.75, 6.0), (9.25, 6.3), (9.85, 5.08))
        elementwise(9.85, 4.85, r"$\odot$")

        # 门内直接显示矩阵乘法；两部分权重并列、偏置相加后合并书写。
        # 重置门仍在旧状态的矩阵乘法与加偏置之后相乘。
        operation(2.70, 1.80,
                  r"$\sigma\!\left(W_\rho\binom{\mathbf{x}_t}{\mathbf{h}_t}+\mathbf{b}_\rho\right)$",
                  width=3.50, height=0.80)
        label(2.70, 2.47, "重置门", size=10)
        curve((4.49, 1.80), (4.65, 1.80), (4.65, 2.47), (4.65, 3.12))
        label(4.86, 2.39, r"$\boldsymbol{\rho}_t$", size=13, ha="left")

        # x_t 紧邻矩阵乘法节点；输入支路不受重置门控制。
        label(6.35, 0.76, r"$\mathbf{x}_t$", size=15)
        arrow_between((6.35, 1.02), (6.35, 1.47))
        operation(6.35, 1.80, r"$W_x\mathbf{x}_t+\mathbf{b}_x$", width=1.88, height=0.60)
        arrow_between((6.35, 2.13), (6.35, 3.12))

        # 更新门分支分别标出 z_t 与 1-z_t，省去仅作减法的独立方框。
        operation(7.25, 4.85,
                  r"$\sigma\!\left(W_z\binom{\mathbf{x}_t}{\mathbf{h}_t}+\mathbf{b}_z\right)$",
                  width=3.50, height=0.80)
        label(7.25, 5.52, "更新门", size=10)
        arrow_between((9.04, 4.85), (9.62, 4.85))
        label(9.40, 5.16, r"$\mathbf{z}_t$", size=12)
        junction(9.23, 4.85)
        curve((9.23, 4.85), (9.66, 4.85), (9.85, 4.12), (9.85, 3.58))
        label(8.95, 4.00, r"$1-\mathbf{z}_t$", size=12)

        # 两条加权状态以短弧汇合；图中不存在状态跨越或边线交叉。
        curve((10.08, 4.85), (10.79, 4.85), (11.55, 4.70), (11.55, 4.33))
        curve((10.08, main_y), (10.79, main_y), (11.55, 3.50), (11.55, 3.87))
        elementwise(11.55, 4.10, r"$+$")
        arrow_between((11.78, 4.10), (12.27, 4.10))
        label(12.76, 4.10, r"$\mathbf{h}_{t+1}$", size=15)

        # 只在图内解释运算符；其余符号定义和公式由正文说明。
        label(9.30, 0.85, r"$\odot$ 逐元素乘    $\oplus$ 逐元素加", size=10)
        save_figure(figure, "gru_architecture")

        # Mermaid 保留相同计算关系；1-z 直接作为分支标签。
        mermaid = """flowchart TB
    subgraph Time[试次间的状态传递]
        hp["h_(t-1)"] --> gp["GRU θ"] --> ht["h_t"] --> gc["GRU θ"] --> hn["h_(t+1)"]
        xp["x_(t-1)"] --> gp
        xt["x_t = (a_t, r_t)"] --> gc
        ht --> readout["线性读出 → softmax"] --> pt["p_t"]
    end
    subgraph Cell[GRU 的内部计算]
        reset["重置门：σ(W_ρ [x_t; h_t] + b_ρ)"]
        update["更新门：σ(W_z [x_t; h_t] + b_z)"]
        %% 圈内点表示逐元素乘，圈内加号表示逐元素加。
        h["h_t"] --> hidden_affine["W_h h_t + b_h"] --> reset_multiply(("·"))
        reset -->|ρ_t| reset_multiply
        x["x_t"] --> input_affine["W_x x_t + b_x"] --> candidate_sum(("+"))
        reset_multiply --> candidate_sum --> tanh["tanh"] --> candidate_multiply(("·"))
        update -->|z_t| old_multiply(("·"))
        h --> old_multiply
        update -->|1 − z_t| candidate_multiply
        candidate_multiply --> final_sum(("+"))
        old_multiply --> final_sum --> result["h_(t+1)"]
    end
    classDef default fill:#fff,stroke:#222,color:#222;
"""
        (OUTPUT_DIR / "gru_architecture.mmd").write_text(mermaid, encoding="utf-8")


def plot_saved_model_results(models, records):
    """复用项目的数值分析与绘图，将两张主图直接导出到章节图片目录。"""
    comparison = summarize_model_comparison(records)
    fields = {}
    for model_name, model in models.items():
        # 与项目 main.py 保持一致；边界只限制显示范围，不改变状态或更新。
        state_limits = (0.0, 1.0) if model_name == "RW" else (-1.0, 1.0)
        fields[model_name] = compute_model_dynamics(
            model,
            records,
            model_name,
            grid_size=41,
            state_limits=state_limits,
            max_arrows_per_condition=None,
            sample_seed=0,
        )
        # None 表示显示全部真实事件；不能意外退回每种条件抽取 100 个箭头。
        field = fields[model_name]
        assert sum(condition["n_displayed"] for condition in field["conditions"]) == field["n_trials"]
        print(f"{model_name} 动力学：{len(records)} 个测试 block，{field['n_trials']} 次更新。")

    # 原绘图函数使用 20 个统一等值层级，直接复用其图形和导出设置。
    # 文稿引用 SVG；另保留项目的 180 dpi PNG 供预览。
    project_plotting.plot_model_comparison(comparison, OUTPUT_DIR)
    project_plotting.plot_dynamics_comparison(fields, OUTPUT_DIR)


def main():
    models, records, _ = load_results(PROJECT_DIR / "results" / "monkey_V_seed0")
    plot_reversal_task()
    plot_rw_architecture()
    plot_gru_architecture()
    plot_saved_model_results(models, records)
    print(f"已保存五组图：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

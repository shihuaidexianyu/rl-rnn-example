"""第一步：读取猴子数据，并按完整 block 整理。

只用两个函数：load_session 整理一个文件，load_blocks 汇总一只猴子的文件。
一个文件对应一个 session，一个 session 包含多个 block。
用一个普通字典表示一个 block，用列表保存多个 block，不需要定义数据集类。

统一的数据约定：
    animal_name: 猴子名称，例如 "V"。
    session_name: 记录日期标识，例如 "V20161005"。
    block_order: 原始 Y 中的 BlockOrder，与 session_name 一起标识一个 block。
    block_type: "what" 或 "where"。
    trial_numbers: 原始块内试次编号，例如 11 到 70，形状为 (T,)。
    actions: 0/1 动作数组，形状为 (T,)。
    rewards: 0/1 奖励数组，形状为 (T,)。

T 表示保留的试次数，默认是 60。
只有 actions 和 rewards 送入模型，其余字段用于核对和追踪数据。
"""

from pathlib import Path

import numpy as np
import scipy.io as sio


def load_session(
    file_path: Path,
    trial_start: int = 11,
    trial_end: int = 70,
) -> list[dict]:
    """读取一个 MAT 文件，返回该 session 的完整 block 列表。

    trial_start、trial_end 使用原始试次编号，两端都包含。
    默认保留第 11–70 次；对应 Python 切片 [10:70]。
    每个 block 包含本文件顶部约定的所有字段，动作与奖励使用 NumPy 数组。
    """
    # TODO 1：检查 1 <= trial_start <= trial_end <= 80。
    if not (1 <= trial_start <= trial_end <= 80):
        raise ValueError(
            "trial_start 和 trial_end 必须满足 1 <= trial_start <= trial_end <= 80。"
        )
    # TODO 2：使用 sio.loadmat(file_path, variable_names=["Y"]) 读取行为数据。
    # 返回的是字典，从中取出 Y；不需要读取神经数据 X。
    # Y 的每一行是一次试次，下面是从 0 开始的列索引：
    # 0：图像选择；1：位置选择；2：奖励；3：试次是否完成。
    # 5：块内试次编号；7：BlockID；8：BlockOrder。
    # 9：真实任务类型，1 为 what、2 为 where；12：block 是否完成。
    beh_data = sio.loadmat(file_path, variable_names=["Y"])["Y"]  # (num_trials, 13)

    # TODO 3：从文件名得到 animal_name 和 session_name。
    # 例如 SPKcounts_V20161005cue_MW_250X250ms.mat 对应 V、V20161005。
    token = file_path.stem.split("_")[1]  # 例如 "V20161005cue"
    if not token.endswith("cue"):
        raise ValueError(f"无法从文件名解析动物和 session：{file_path.name}")
    animal_name = token[0]
    session_name = token.removesuffix("cue")

    # TODO 4：在当前文件内按 BlockOrder 分组，保持原始出现顺序。
    # BlockID 是实验条件编号，不能单独作为 block 的唯一标识。
    block_order_col = beh_data[:, 8].astype(int)
    unique_orders, first_indices = np.unique(block_order_col, return_index=True)
    unique_orders = unique_orders[np.argsort(first_indices)]

    blocks = []
    for block_order in unique_orders:
        rows = beh_data[block_order_col == block_order]
        # TODO 5：保留 BlockID 在 1–24 且 BlockCompleted=1 的完整 block。
        # 核对每组恰好 80 条、试次编号为 1–80、任务类型在组内一致。
        # 数据不符合这些约定时，应明确报错，避免静默混合不同 block。
        block_id = int(rows[0, 7])
        if not (1 <= block_id <= 24):
            continue
        if not np.all(rows[:, 12] == 1):
            continue
        if len(rows) != 80:
            raise ValueError(f"Block {block_order} does not have exactly 80 trials.")
        trial_numbers = rows[:, 5].astype(int)
        if not np.array_equal(trial_numbers, np.arange(1, 81)):
            raise ValueError(f"Block {block_order} has non-consecutive trial numbers.")
        task_types = np.unique(rows[:, 9].astype(int))
        if len(task_types) > 1:
            raise ValueError(f"Block {block_order} has inconsistent task types.")
        task_type = int(task_types[0])
        if task_type not in (1, 2):
            raise ValueError(f"Block {block_order} has unknown task type {task_type}.")
        block_type = "what" if task_type == 1 else "where"
        # TODO 6：what 使用图像选择作为 actions；where 使用位置选择。
        action_col = 0 if task_type == 1 else 1
        actions = rows[:, action_col].astype(int)
        rewards = rows[:, 2].astype(int)
        # TODO 7：按块内试次编号保留指定范围，同步截取动作与奖励。
        # 检查动作和奖励均为 0/1，且当前保留的记录都是已完成试次。
        # 将 actions 转成整数数组，之后才能用作动作索引和分类标签。
        mask = (trial_numbers >= trial_start) & (trial_numbers <= trial_end)
        trial_numbers = trial_numbers[mask]
        actions = actions[mask]
        rewards = rewards[mask]
        completed = rows[mask, 3].astype(int)
        if not np.all((actions == 0) | (actions == 1)):
            raise ValueError(f"Block {block_order} has invalid actions.")
        if not np.all((rewards == 0) | (rewards == 1)):
            raise ValueError(f"Block {block_order} has invalid rewards.")
        if not np.all(completed == 1):
            raise ValueError(f"Block {block_order} has unfinished trials.")
        # TODO 8：每组整理成一个字典，返回字典列表。
        # 不要将不同 block 拼成一条时间序列。
        blocks.append(
            {
                "animal_name": animal_name,
                "session_name": session_name,
                "block_order": int(block_order),
                "block_type": block_type,
                "trial_numbers": trial_numbers,
                "actions": actions,
                "rewards": rewards,
            }
        )
    return blocks


def load_blocks(
    data_dir: Path,
    animal_name: str,
    trial_start: int = 11,
    trial_end: int = 70,
) -> list[dict]:
    """读取同一只猴子的多个 MAT 文件，返回统一的 block 列表。

    当前目录的八个文件中，V 应有 96 个完整 block，W 应有 94 个。
    默认每个 block 保留 60 次试次；这里只使用行为数据，不使用神经数据 X。
    """
    # TODO 1：检查 animal_name 只能是 "V" 或 "W"。
    if animal_name not in ["V", "W"]:
        raise ValueError(f"Invalid animal_name: {animal_name}. Must be 'V' or 'W'.")
    # TODO 2：只查找所选猴子的 SPKcounts_*cue_MW_250X250ms.mat 文件。
    # 对文件名排序，保证多次运行的 block 顺序一致；找不到文件时明确报错。
    # 动物名和 cue 之间还有日期，glob 模式里需要通配符。
    mat_files = sorted(data_dir.glob(f"SPKcounts_{animal_name}*cue_MW_250X250ms.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No MAT files found for animal {animal_name}.")
    # TODO 3：逐个调用 load_session，传入相同的 trial_start 和 trial_end。
    all_blocks = []
    for mat_file in mat_files:
        all_blocks.extend(
            load_session(mat_file, trial_start=trial_start, trial_end=trial_end)
        )
    # TODO 4：合并返回的列表，并检查 (session_name, block_order) 没有重复。
    seen = set()
    for block in all_blocks:
        key = (block["session_name"], block["block_order"])
        if key in seen:
            raise ValueError(f"Duplicate block found: {key}")
        seen.add(key)
    return all_blocks

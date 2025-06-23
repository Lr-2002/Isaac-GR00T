import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os

# 数据映射配置
state_indices = {
    "qpos_left_arm": (0, 7),
    "qpos_right_arm": (7, 14),
    "qvel_left_arm": (14, 21),
    "qvel_right_arm": (21, 28),
    "ee_left_arm": (28, 35),
    "ee_right_arm": (35, 42),
    "torque_left_arm": (42, 49),
    "torque_right_arm": (49, 56)
}
action_indices = {
    "action_left_arm": (0, 7),
    "action_right_arm": (7, 14)
}

def convert_parquet(file_path, save_path=None):
    df = pd.read_parquet(file_path)
    n = len(df)

    # 1. 提取 observation.state
    state = df.iloc[:, 0:56].values  # 假设前 56 列为 state

    # 2. 提取 action
    action = df.iloc[:, 56:70].values  # 假设 action 是 14 维，从第 56 到 69 列

    # 3. timestamp
    timestamp = np.arange(n).reshape(-1, 1) / 30.0

    # 4. index
    index = np.arange(n).reshape(-1, 1)

    # 5. reward
    reward = np.zeros((n, 1))
    reward[-1] = 1

    # 6. done
    done = np.zeros((n, 1), dtype=bool)
    done[-1] = True

    # 7. 构造最终 DataFrame 或 dict
    result = {
        "observation.state": state,
        "action": action,
        "timestamp": timestamp,
        "index": index,
        "next.reward": reward,
        "next.done": done
    }

    final_df = pd.DataFrame({
        "observation.state": list(state),
        "action": list(action),
        "timestamp": timestamp.flatten(),
        "index": index.flatten(),
        "next.reward": reward.flatten(),
        "next.done": done.flatten(),
        "annotation.human.action.task_description": 0,
    })

    if save_path is not None:
        final_df.to_parquet(save_path, index=False)
    print(final_df)
    return final_df

# 示例用法
# convert_parquet('/home/a4060/project/data/fold_clothes/gr00t/data/chunk-000/episode_000000.parquet', './test.parquet')

# 如果你想批量处理某个文件夹下的所有 parquet 文件：
import glob
input_dir = "/home/a4060/project/data/fold_clothes/gr00t/chunk-000_bak"
output_dir = "/home/a4060/project/data/fold_clothes/gr00t/data/chunk-000"
os.makedirs(output_dir, exist_ok=True)
for file in glob.glob(os.path.join(input_dir, "*.parquet")):
    name = os.path.basename(file)
    convert_parquet(file, os.path.join(output_dir, name))
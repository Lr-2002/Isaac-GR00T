import pandas as pd

# 加载 Parquet 文件
file_path = '/home/a4060/project/Isaac-GR00T/demo_data/robot_sim.PickNPlace/data/chunk-000/episode_000001.parquet'  # 替换为你的 Parquet 文件路径
df = pd.read_parquet(file_path)

# 输出表的前 5 行信息
print(df.head())
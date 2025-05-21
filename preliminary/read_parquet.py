import pandas as pd

# 设置显示所有列和不限制列宽
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# 读取 train_easy.parquet 文件
df = pd.read_parquet("/data/zliu331/temporal_reasoning/TinyZero/datasets/train_nyt.parquet")

# 显示前 5 行数据
print(df.head())

# # 使用 to_string() 输出完整信息
# print(df.to_string())
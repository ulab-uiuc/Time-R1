# import pandas as pd

# # 读取原始数据集
# df = pd.read_parquet('/data/zliu331/temporal_reasoning/TinyZero/datasets/test_nyt.parquet')
# print("原始数据集的数量：", len(df))

# # 使用固定随机种子1024，随机抽取1024条数据
# small_df = df.sample(n=1024, random_state=1024)
# print("抽样后的数据集数量：", len(small_df))

# # 保存为新的Parquet文件
# small_df.to_parquet('/data/zliu331/temporal_reasoning/TinyZero/datasets/small_test_nyt.parquet')

# print("生成 small_test_nyt.parquet 数据集成功！")






# import pandas as pd
# from datetime import datetime

# def parse_year_month(date_str):
#     """
#     date_str is expected to be 'YYYY-MM'.
#     Returns (year, month) as integers.
#     If parsing fails, returns (None, None).
#     """
#     try:
#         dt = datetime.strptime(date_str.strip(), "%Y-%m")
#         return dt.year, dt.month
#     except:
#         return None, None

# # 1. Load the original data
# df = pd.read_parquet("/data/zliu331/temporal_reasoning/TinyZero/datasets/test_nyt.parquet")

# # 2. Extract "YYYY-MM" from reward_model.ground_truth.true_pub_date
# #    so it's easier to filter by year/month.
# def get_true_date(row):
#     """
#     Each row['reward_model'] is a dict of the form:
#        {"style": "...", "ground_truth": {"true_pub_date": "YYYY-MM"}}
#     """
#     return row["reward_model"]["ground_truth"]["true_pub_date"]

# df["date_str"] = df.apply(get_true_date, axis=1)

# # 3. Parse out year/month into new columns
# df["year_month"] = df["date_str"].apply(parse_year_month)
# df["year"] = df["year_month"].apply(lambda x: x[0])
# df["month"] = df["year_month"].apply(lambda x: x[1])

# # 4. Create two filters:
# #    - Filter A: 2024-01 <= date <= 2024-06  (for new training set)
# #    - Filter B: 2024-07 <= date <= 2025-02  (for sampling validation)
# train_mask = (
#     (df["year"] == 2024) &
#     (df["month"] >= 1) &
#     (df["month"] <= 6)
# )

# val_candidate_mask = (
#     ((df["year"] == 2024) & (df["month"] >= 7)) |
#     ((df["year"] == 2025) & (df["month"] <= 2))
# )

# # 5. Extract the subsets
# train_df = df[train_mask].copy()
# val_candidates = df[val_candidate_mask].copy()

# # 6. Randomly sample 1024 rows for validation (with a fixed seed).
# val_df = val_candidates.sample(n=1024, random_state=1024)

# # 7. Save results to new parquet files
# train_df.to_parquet("/data/zliu331/temporal_reasoning/TinyZero/datasets/train_2024JanToJun.parquet", index=False)
# val_df.to_parquet("/data/zliu331/temporal_reasoning/TinyZero/datasets/val_2024JulTo2025Feb_1024.parquet", index=False)

# print(f"Training set size: {len(train_df)}")
# print(f"Validation set size: {len(val_df)}")
# print("Done!")





# import pandas as pd
# from datetime import datetime

# def parse_year_month(date_str):
#     """
#     date_str is expected to be 'YYYY-MM'.
#     Returns (year, month) as integers.
#     If parsing fails, returns (None, None).
#     """
#     try:
#         dt = datetime.strptime(date_str.strip(), "%Y-%m")
#         return dt.year, dt.month
#     except:
#         return None, None

# # 1. Load the original data
# df = pd.read_parquet("/data/zliu331/temporal_reasoning/TinyZero/datasets/test_nyt.parquet")

# # 2. Extract "YYYY-MM" from reward_model.ground_truth.true_pub_date
# #    so it's easier to filter by year/month.
# def get_true_date(row):
#     """
#     Each row['reward_model'] is a dict of the form:
#        {"style": "...", "ground_truth": {"true_pub_date": "YYYY-MM"}}
#     """
#     return row["reward_model"]["ground_truth"]["true_pub_date"]

# df["date_str"] = df.apply(get_true_date, axis=1)

# # 3. Parse out year/month into new columns
# df["year_month"] = df["date_str"].apply(parse_year_month)
# df["year"] = df["year_month"].apply(lambda x: x[0])
# df["month"] = df["year_month"].apply(lambda x: x[1])

# # 4. Create two filters:
# #    - Filter A: 2024-01 <= date <= 2024-12  (for new training set)
# #    - Filter B: 2025-01 <= date <= 2025-02  (for sampling validation)
# train_mask = (
#     (df["year"] == 2024) &
#     (df["month"] >= 1) &
#     (df["month"] <= 12)
# )

# val_candidate_mask = (
#     (df["year"] == 2025) &
#     (df["month"] >= 1) &
#     (df["month"] <= 2)
# )

# # 5. Extract the subsets
# train_df = df[train_mask].copy()
# val_candidates = df[val_candidate_mask].copy()

# # 6. Randomly sample 1024 rows for validation (with a fixed seed).
# val_df = val_candidates.sample(n=1024, random_state=1024)

# # 7. Save results to new parquet files
# train_df.to_parquet("/data/zliu331/temporal_reasoning/TinyZero/datasets/train_2024JanToDec.parquet", index=False)
# val_df.to_parquet("/data/zliu331/temporal_reasoning/TinyZero/datasets/val_2025JanToFeb_1024.parquet", index=False)

# print(f"Training set size: {len(train_df)}")
# print(f"Validation set size: {len(val_df)}")
# print("Done!")






import pandas as pd
# from datetime import datetime

# --- Combine old data (random sample) with new data (2024Jan-Jun) ---
print("\nNow sampling 11909 rows from train_easy_nyt.parquet and merging with new 2024 data...")

# easy_df = pd.read_parquet("/data/zliu331/temporal_reasoning/TinyZero/datasets/train_easy_nyt.parquet")
other_df = pd.read_parquet("/data/zliu331/temporal_reasoning/TinyZero/datasets/train_nyt.parquet")
# Sample 11909 rows from the old dataset
# sample_easy_df = easy_df.sample(n=11909, random_state=1024)
sample_other_df = other_df.sample(n=24570, random_state=1024)

# Read the new 2024JanToJun data
new_df = pd.read_parquet("/data/zliu331/temporal_reasoning/TinyZero/datasets/train_2024JanToDec.parquet")

# Combine them
combined_df = pd.concat([sample_other_df, new_df], ignore_index=True)

# Shuffle the combined dataset
combined_df = combined_df.sample(frac=1.0, random_state=1024).reset_index(drop=True)

# Save the final dataset
mixed_output_path = "/data/zliu331/temporal_reasoning/TinyZero/datasets/train_mixed_hard_2024JanToDec.parquet"
combined_df.to_parquet(mixed_output_path, index=False)

print(f"Created combined dataset: {len(combined_df)} rows.")
print(f"Saved to {mixed_output_path}")





# import pandas as pd

# # --- Combine all original data with new data (2024Jan-Dec) ---
# print("\nNow combining all rows from train_easy_nyt.parquet with new 2024 data (Jan-Dec)...")

# easy_df = pd.read_parquet("/data/zliu331/temporal_reasoning/TinyZero/datasets/train_easy_nyt.parquet")
# # Use all rows from the original data (no sampling)

# # Read the new 2024JanToDec data
# new_df = pd.read_parquet("/data/zliu331/temporal_reasoning/TinyZero/datasets/train_2024JanToDec.parquet")

# # Combine them
# combined_df = pd.concat([easy_df, new_df], ignore_index=True)

# # Shuffle the combined dataset
# combined_df = combined_df.sample(frac=1.0, random_state=1024).reset_index(drop=True)

# # Save the final dataset
# mixed_output_path = "/data/zliu331/temporal_reasoning/TinyZero/datasets/train_mixed_2024JanToDec.parquet"
# combined_df.to_parquet(mixed_output_path, index=False)

# print(f"Created combined dataset: {len(combined_df)} rows.")
# print(f"Saved to {mixed_output_path}")
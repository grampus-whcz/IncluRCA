import pandas as pd

# 文件路径
file_path = "/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phasetwo/2025-06-19/cloudbed/log-parquet/log_filebeat-server_2025-06-19_02-00-00.parquet"

# 读取 Parquet 文件
df = pd.read_parquet(file_path)

# 定义要搜索的字符串
search_str = "tidb-tidb-0"  # ←←← 替换为你想搜索的内容

# 在所有字符串类型的列中搜索（避免对数值列做 str.contains）
text_columns = df.select_dtypes(include=['object']).columns

# 创建一个布尔掩码，只要任意文本列包含该字符串就算匹配
mask = pd.Series([False] * len(df))
for col in text_columns:
    mask |= df[col].astype(str).str.contains(search_str, na=False)

# 打印匹配的行
matching_rows = df[mask]
if not matching_rows.empty:
    print(f"Found {len(matching_rows)} row(s) containing '{search_str}':\n")
    for idx, row in matching_rows.iterrows():
        print(row.to_dict())  # 或者 print(row) 以表格形式
else:
    print(f"No rows found containing '{search_str}'.")
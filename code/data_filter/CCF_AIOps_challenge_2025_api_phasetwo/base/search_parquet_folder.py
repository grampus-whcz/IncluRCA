import os
import pandas as pd

# === 配置区 ===
search_dir = "/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phasetwo/2025-06-27/cloudbed/trace-parquet"
search_str = "tikv"  # ←←← 替换为你想搜索的字符串
# ==============

parquet_files = [f for f in os.listdir(search_dir) if f.endswith('.parquet')]
parquet_files.sort()  # 可选：按文件名排序

found_any = False

for filename in parquet_files:
    filepath = os.path.join(search_dir, filename)
    print(f"\n🔍 Scanning {filename} ...")

    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        continue

    if df.empty:
        print("  (empty file)")
        continue

    # 仅选择 object（字符串）类型的列进行搜索
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) == 0:
        print("  (no text columns to search)")
        continue

    # 构建掩码：任意文本列包含 search_str
    mask = pd.Series([False] * len(df))
    for col in text_cols:
        # 转为字符串并搜索，na=False 避免 NaN 报错
        mask |= df[col].astype(str).str.contains(search_str, na=False, case=True)

    matches = df[mask]
    if not matches.empty:
        found_any = True
        print(f"  ✅ Found {len(matches)} match(es):")
        for idx, row in matches.iterrows():
            # 打印整行（转为字典便于阅读）
            print(f"    → {row.to_dict()}")
    else:
        print("  (no matches)")

if not found_any:
    print(f"\n🔎 No matches found for '{search_str}' in any .parquet file.")
else:
    print(f"\n🎉 Search completed. Matches found for '{search_str}'.")
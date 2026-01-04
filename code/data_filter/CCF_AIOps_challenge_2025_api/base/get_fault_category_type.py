import json
from collections import defaultdict
import os

# 文件路径
# file_path = '/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/answer/phase1.jsonl'
file_path = '/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/answer/phase2.jsonl'

# 用于存储结果：fault_category -> set of fault_type
category_to_types = defaultdict(set)

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"❌ 文件不存在: {file_path}")
else:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_number = 0
            for line in f:
                line_number += 1
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                try:
                    data = json.loads(line)
                    fault_category = data.get("fault_category")
                    fault_type = data.get("fault_type")

                    if fault_category and fault_type:
                        category_to_types[fault_category].add(fault_type)
                    else:
                        print(f"⚠️ 第 {line_number} 行缺少 fault_category 或 fault_type: {data}")

                except json.JSONDecodeError as e:
                    print(f"❌ 第 {line_number} 行 JSON 解析错误: {e}")
                    continue

        # 输出结果
        print("\n🔍 故障类别（fault_category）及其包含的故障类型（fault_type）：\n")
        for category in sorted(category_to_types.keys()):
            types = sorted(category_to_types[category])
            print(f"📁 {category}:")
            for t in types:
                print(f"    ├── {t}")
            print()

    except Exception as e:
        print(f"❌ 读取文件时发生异常: {e}")
        
        
# 🔍 phase1 故障类别（fault_category）及其包含的故障类型（fault_type）：

# 📁 jvm fault:
#     ├── jvm cpu
#     ├── jvm exception
#     ├── jvm gc
#     ├── jvm latency

# 📁 network attack:
#     ├── network corrupt
#     ├── network delay
#     ├── network loss

# 📁 node fault:
#     ├── node cpu
#     ├── node disk fill
#     ├── node memory

# 📁 pod fault:
#     ├── pod failure
#     ├── pod kill

# 📁 stress test:
#     ├── cpu stress
#     ├── memory stress


# 🔍 phase2 故障类别（fault_category）及其包含的故障类型（fault_type）：

# 📁 dns fault:
#     ├── dns error

# 📁 erroneous change:
#     ├── code error

# 📁 io fault:
#     ├── io fault

# 📁 misconfiguration:
#     ├── target port misconfig

# 📁 network attack:
#     ├── network corrupt
#     ├── network delay
#     ├── network loss

# 📁 node fault:
#     ├── node cpu
#     ├── node disk fill
#     ├── node memory

# 📁 pod fault:
#     ├── pod failure

# 📁 stress test:
#     ├── cpu stress
#     ├── memory stress
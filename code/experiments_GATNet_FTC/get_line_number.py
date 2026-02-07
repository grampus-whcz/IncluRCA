# sed -n '122662,122678p' experiments_a.log


# def find_line_after_evaluation_dataset(filename):
#     target_line = None
#     found_line_numbers = []

#     with open(filename, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     for i, line in enumerate(lines):
#         if "evaluation dataset type: test" in line:
#             target_line = i  # 保存匹配行的行号
#             # 检查下一行是否存在并包含目标字符串
#             if target_line + 1 < len(lines):
#                 next_line = lines[target_line + 1]
#                 if "node    precision | micro: 0.69" in next_line:
#                     found_line_numbers.append(target_line + 1)  # 返回下一行的行号（从0开始）

#     return found_line_numbers

# # 使用示例
# filename = "experiments_a.log"  # 替换为你的文件名
# result = find_line_after_evaluation_dataset(filename)

# if result:
#     print("匹配的下一行行号为（从0开始）：", result)
# else:
#     print("未找到匹配内容")

import re
from collections import defaultdict

def find_line_after_evaluation_dataset(filename):
    precision_dict = defaultdict(list)

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "evaluation dataset type: test" in line:
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # 匹配 node    precision | micro: 后的浮点数
                match = re.search(r"node\s+precision \| micro:\s+(\d*\.\d+)", next_line)
                if match:
                    precision_str = match.group(1)
                    try:
                        precision_val = float(precision_str)
                        # 截断到小数点后两位，例如 0.674419 → 0.67
                        truncated = float(f"{precision_val:.2f}")
                        precision_dict[str(truncated)].append(i + 1)
                    except ValueError:
                        continue  # 忽略非法数值

    return precision_dict

# 使用示例
filename = "/root/shared-nvme/work/code/RCA/IncluRCA/code/experiments_a_MSCSEAttention_conv_fc_FTC.log"  # 替换为你的文件名
result = find_line_after_evaluation_dataset(filename)

# 输出结果
if result:
    for key in sorted(result.keys(), key=float):  # 按浮点数值排序
        print(f"micro: {key} 区间，匹配的行号（从0开始）：{result[key]}")
else:
    print("未找到匹配内容")
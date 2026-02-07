import re

# 定义目标训练序号（从1开始计数）
target_indices = [
    48, 62, 63, 65, 72, 73, 74, 78, 85, 86, 87, 89, 92, 96, 97,
    102, 104, 105, 107, 108, 115, 119, 121, 123, 124, 125, 133,
    135, 137, 140, 142, 143, 145, 146, 148, 151, 152, 157, 160,
    162, 164, 170, 186, 190, 192
]

# 将索引转为集合便于快速查找（注意：日志块索引从0开始，所以减1）
target_set = set(idx - 1 for idx in target_indices)

# 日志文件路径
log_file = "/root/shared-nvme/work/code/RCA/IncluRCA/code/experiments_a_MSCSEAttention_FTC.log"

# 正则：匹配一次完整训练的结尾（从 "evaluation dataset type: test" 到 "✅ Finished: ..."）
pattern = re.compile(
    r'(202\d-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO - ----------\n'
    r'202\d-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO - evaluation dataset type: test.*?'
    r'✅ Finished: (.*?) → log saved to .*?)\n\n',
    re.DOTALL
)

# 读取整个日志
with open(log_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 提取所有训练结尾块
matches = pattern.findall(content)
blocks = [m[0] for m in matches]  # m[0] 是完整块，m[1] 是 config_name
config_names = [m[1] for m in matches]

print("Extracted {} training blocks.".format(len(blocks)))

# 检查是否足够
if len(blocks) < max(target_indices):
    print(f"Warning: Only {len(blocks)} blocks found, but max index is {max(target_indices)}")

# 存储结果
results = []

# 遍历所有块，只处理目标索引
for i, (block, config_name) in enumerate(zip(blocks, config_names)):
    if i not in target_set:
        continue

    # 从 block 中提取 GAT 和 activation 信息
    gat1_match = re.search(r'GAT_name1:\s+(GATConv|GATv2Conv)', block)
    gat2_match = re.search(r'GAT_name2:\s+(GATConv|GATv2Conv)', block)
    act1_match = re.search(r'activ_fun1:\s+([a-z0-9_]+)', block)
    act2_match = re.search(r'activ_fun2:\s+([a-z0-9_]+)', block)

    if not (gat1_match and gat2_match and act1_match and act2_match):
        print(f"Warning: Failed to parse block {i+1}")
        continue

    gat1 = gat1_match.group(1)
    gat2 = gat2_match.group(1)
    act1 = act1_match.group(1)
    act2 = act2_match.group(1)

    # 构造 value 字符串
    value_str = f"{gat1} {gat2} {act1} {act2}"

    # 构造 key（即 config_name）
    key = config_name  # e.g., "max_conv_GATv2Conv_GATv2Conv_relu6_relu6"

    results.append((key, value_str))

# 按原始 target_indices 顺序排序（因为 regex 匹配是顺序的，但保险起见）
# 我们记录每个 result 对应的原始索引
indexed_results = []
for i, (block, config_name) in enumerate(zip(blocks, config_names)):
    if i in target_set:
        gat1_match = re.search(r'GAT_name1:\s+(GATConv|GATv2Conv)', block)
        gat2_match = re.search(r'GAT_name2:\s+(GATConv|GATv2Conv)', block)
        act1_match = re.search(r'activ_fun1:\s+([a-z0-9_]+)', block)
        act2_match = re.search(r'activ_fun2:\s+([a-z0-9_]+)', block)
        if all([gat1_match, gat2_match, act1_match, act2_match]):
            gat1 = gat1_match.group(1)
            gat2 = gat2_match.group(1)
            act1 = act1_match.group(1)
            act2 = act2_match.group(1)
            value_str = f"{gat1} {gat2} {act1} {act2}"
            indexed_results.append((i, config_name, value_str))

# 按 target_indices 顺序输出
output_lines = []
for idx in target_indices:
    block_index = idx - 1
    found = False
    for i, key, val in indexed_results:
        if i == block_index:
            output_lines.append(f'    ["{key}"]="{val}"')
            found = True
            break
    if not found:
        output_lines.append(f'    # MISSING: training #{idx}')

# 打印最终结果
for line in output_lines:
    print(line)
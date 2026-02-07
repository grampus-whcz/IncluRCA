import re
import json
import os

def parse_log_file(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()

    results = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 寻找评估开始标记
        if "evaluation dataset type: test" in line:
            metrics = {}
            i += 1
            # 读取直到下一个 ----------
            while i < len(lines) and "----------" not in lines[i]:
                l = lines[i].strip()
                # 匹配如: node    precision | micro: 0.655914; ...
                match = re.search(r'.*(node|service|pod)\s*(precision|recall|f1)\s*\|\s*micro:\s*([0-9.]+);.*', l)
                if match:
                    entity, metric, value = match.groups()
                    key = f"{entity}_{metric}_micro"
                    metrics[key] = float(value)
                i += 1

            # 现在寻找紧随其后的 GATNet 配置
            config = {}
            # 找到 ----GATNet---- 行
            while i < len(lines) and not lines[i].startswith("----GATNet----"):
                i += 1

            if i < len(lines) and lines[i].startswith("----GATNet----"):
                i += 1  # 跳过 ----GATNet---- 行
                # 只读接下来的最多 4 行（或提前遇到空行就停）
                for _ in range(4):
                    if i >= len(lines):
                        break
                    l = lines[i].strip()
                    if l == "":
                        break
                    if ":" in l:
                        key, val = l.split(":", 1)
                        config[key.strip()] = val.strip()
                    i += 1

            # 保存这一轮的结果+配置
            results.append({
                'metrics': metrics,
                'config': config
            })
        else:
            i += 1

    print(f"总行数: {len(lines)}")
    test_count = sum(1 for line in lines if "evaluation dataset type: test" in line)
    print(f"发现 {test_count} 个 test block")

    return results


def filter_configs(results):
    filtered = []
    for entry in results:
        m = entry['metrics']
        cond = (
            m.get('node_precision_micro', 0) >= 0.66 and
            m.get('node_recall_micro', 0) >= 0.9 and
            m.get('node_f1_micro', 0) >= 0.77 and

            m.get('service_precision_micro', 0) >= 0.72 and
            m.get('service_recall_micro', 0) >= 0.7 and
            m.get('service_f1_micro', 0) >= 0.74 and

            m.get('pod_precision_micro', 0) >= 0.66 and
            m.get('pod_recall_micro', 0) >= 0.8 and
            m.get('pod_f1_micro', 0) >= 0.73
        )
        if cond:
            filtered.append(entry)
    return filtered


def main():
    log_file = "/root/shared-nvme/work/code/RCA/IncluRCA/code/experiments_a_MSCSEAttention_conv_fc_FTC.log"
    output_json = "/root/shared-nvme/work/code/RCA/IncluRCA/code/experiments_a_MSCSEAttention_conv_fc_FTC_good_configs.json"

    results = parse_log_file(log_file)
    good_configs = filter_configs(results)

    print(f"共找到 {len(good_configs)} 个满足条件的配置：\n")
    for idx, entry in enumerate(good_configs, 1):
        print(f"=== 配置 #{idx} ===")
        for k, v in entry['config'].items():
            print(f"{k}: {v}")
        m = entry['metrics']
        print("对应指标:")
        for key in ['node', 'service', 'pod']:
            p = m.get(f'{key}_precision_micro', 'N/A')
            r = m.get(f'{key}_recall_micro', 'N/A')
            f1 = m.get(f'{key}_f1_micro', 'N/A')
            if isinstance(p, float):
                p = f"{p:.6f}"
            if isinstance(r, float):
                r = f"{r:.6f}"
            if isinstance(f1, float):
                f1 = f"{f1:.6f}"
            print(f"  {key} - P: {p}, R: {r}, F1: {f1}")
        print()

    # 保存到 JSON 文件（确保数值可序列化）
    serializable_configs = []
    for entry in good_configs:
        serializable_configs.append({
            'config': entry['config'],
            'metrics': {k: float(v) for k, v in entry['metrics'].items()}
        })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(serializable_configs, f, indent=4, ensure_ascii=False)

    print(f"✅ 满足条件的配置已保存至: {output_json}")


if __name__ == "__main__":
    main()
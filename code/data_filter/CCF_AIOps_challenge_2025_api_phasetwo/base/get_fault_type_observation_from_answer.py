import json
import os
from collections import defaultdict

# 文件路径
# file_path = '/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/answer/phase1.jsonl'
file_path = '/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/answer/phase2.jsonl'

# 存储结果：fault_type -> { metric: set, log: set, trace: set }
fault_type_to_signals = defaultdict(lambda: {
    'metric': set(),
    'log': set(),
    'trace': set()
})

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"❌ 文件不存在: {file_path}")
else:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)

                    fault_type = data.get("fault_type")
                    if not fault_type:
                        print(f"⚠️ 第 {line_num} 行缺少 fault_type")
                        continue

                    # 获取信号容器
                    signals = fault_type_to_signals[fault_type]

                    # === 1. 解析 key_observation（通常是 metric 列表）===
                    key_observation = data.get("key_observation", {})
                    if isinstance(key_observation, dict):
                        metric_list = key_observation.get("metric", [])
                        upstream_metric_list = key_observation.get("upstream_metric", [])
                        all_metrics = metric_list + upstream_metric_list
                        signals['metric'].update(all_metrics)

                    # === 2. 解析 key_observations（列表，含 type）===
                    key_observations = data.get("key_observations", [])
                    if isinstance(key_observations, list):
                        for item in key_observations:
                            if not isinstance(item, dict):
                                continue
                            item_type = item.get("type")
                            keywords = item.get("keyword", [])
                            if not isinstance(keywords, list):
                                keywords = [keywords]

                            if item_type == "metric":
                                # 可选：提取 category 信息，但这里只关心 keyword
                                signals['metric'].update(keywords)
                            elif item_type == "log":
                                signals['log'].update(keywords)
                            elif item_type == "trace":
                                # trace 的 keyword 是服务实例名，但 subtype 是关键
                                trace_subtype = item.get("subtype")
                                if trace_subtype:
                                    signals['trace'].add(trace_subtype)
                                # signals['trace'].update(keywords)  # 有时 keyword 包含实例名，也可保留

                except json.JSONDecodeError as e:
                    print(f"❌ 第 {line_num} 行 JSON 解析错误: {e}")
                    continue

        # === 输出结果 ===
        print("\n📊 每个 fault_type 关联的诊断信号（metric / log / trace）：\n")
        for fault_type in sorted(fault_type_to_signals.keys()):
            signals = fault_type_to_signals[fault_type]
            print(f"🔍 fault_type: {fault_type}")
            if signals['metric']:
                m_str = ", ".join(sorted(signals['metric']))
                print(f"   ├── metric: {m_str}")
            else:
                print(f"   ├── metric: (无)")
            if signals['trace']:
                t_str = ", ".join(sorted(signals['trace']))
                print(f"   ├── trace: {t_str}")
            else:
                print(f"   ├── trace: (无)")
            if signals['log']:
                l_str = ", ".join(sorted(signals['log']))
                print(f"   └── log: {l_str}")
            else:
                print(f"   └── log: (无)")
            print()

    except Exception as e:
        print(f"❌ 读取文件时发生异常: {e}")
        
        
        
# 📊 phase1 每个 fault_type 关联的诊断信号（metric / log / trace）：

# 🔍 fault_type: cpu stress
#    ├── metric: max_rrt+, pod_cpu_usage, pod_processes, request-, response-, rrt, rrt+, timeout+
#    ├── trace: latency_anomalies
#    └── log: (无)

# 🔍 fault_type: jvm cpu
#    ├── metric: client_error+, max_rrt+, pod_cpu_usage, pod_network_transmit_packets, rrt, rrt+
#    ├── trace: latency_anomalies
#    └── log: error, exception, failed

# 🔍 fault_type: jvm exception
#    ├── metric: (无)
#    ├── trace: (无)
#    └── log: error, exception, failed, stall

# 🔍 fault_type: jvm gc
#    ├── metric: (无)
#    ├── trace: (无)
#    └── log: error, exception, failed, stall

# 🔍 fault_type: jvm latency
#    ├── metric: client_error_ratio, error_ratio, rrt
#    ├── trace: latency_anomalies
#    └── log: error, exception, failed

# 🔍 fault_type: memory stress
#    ├── metric: client_error+, pod_network_receive_bytes, pod_network_receive_packets, pod_network_transmit_bytes, pod_network_transmit_packets, pod_processes, request, response
#    ├── trace: latency_anomalies, request_proportion_anomalies
#    └── log: error

# 🔍 fault_type: network corrupt
#    ├── metric: client_error_ratio, error_ratio, max_rrt+, request-, response-, rrt, rrt+, timeout, timeout+
#    ├── trace: latency_anomalies, request_proportion_anomalies
#    └── log: error, exception, failed, timeout

# 🔍 fault_type: network delay
#    ├── metric: client_error+, max_rrt+, request-, response-, rrt, rrt+
#    ├── trace: latency_anomalies
#    └── log: error, exception, failed

# 🔍 fault_type: network loss
#    ├── metric: max_rrt+, request-, response-, rrt, rrt+
#    ├── trace: latency_anomalies, request_proportion_anomalies
#    └── log: abort, disconnect, error, exception, failed, retry, timeout

# 🔍 fault_type: node cpu
#    ├── metric: node_cpu_usage_rate
#    ├── trace: (无)
#    └── log: (无)

# 🔍 fault_type: node disk fill
#    ├── metric: node_cpu_usage_rate, node_filesystem_usage_rate
#    ├── trace: (无)
#    └── log: (无)

# 🔍 fault_type: node memory
#    ├── metric: node_memory_usage_rate
#    ├── trace: latency_anomalies
#    └── log: error, failed

# 🔍 fault_type: pod failure
#    ├── metric: error+, max_rrt+, request-, response-, rrt, rrt+, server_error+, timeout, timeout+
#    ├── trace: latency_anomalies, request_proportion_anomalies
#    └── log: error, exception, failed, refused, unavailable

# 🔍 fault_type: pod kill
#    ├── metric: client_error+, request-, response-, timeout+
#    ├── trace: latency_anomalies, request_proportion_anomalies
#    └── log: (无)


# 📊 phase2 每个 fault_type 关联的诊断信号（metric / log / trace）：

# 🔍 fault_type: code error
#    ├── metric: client_error_ratio, error_ratio, rrt, timeout
#    ├── trace: latency_anomalies, request_proportion_anomalies
#    └── log: error, exception, failed

# 🔍 fault_type: cpu stress
#    ├── metric: pod_cpu_usage, pod_processes, rrt, timeout
#    ├── trace: latency_anomalies
#    └── log: error, exception

# 🔍 fault_type: dns error
#    ├── metric: error_ratio, rrt, server_error_ratio
#    ├── trace: latency_anomalies
#    └── log: error, exception, failed, misbehaving, timeout, unavailable

# 🔍 fault_type: io fault
#    ├── metric: tikv_cpu_usage, tikv_read_mbps, tikv_snapshot_apply_count, tikv_write_wal_mbps
#    ├── trace: latency_anomalies
#    └── log: (无)

# 🔍 fault_type: memory stress
#    ├── metric: client_error_ratio, error_ratio
#    ├── trace: (无)
#    └── log: error, exception, failed

# 🔍 fault_type: network corrupt
#    ├── metric: rrt
#    ├── trace: latency_anomalies, request_proportion_anomalies
#    └── log: error, exception, failed, timeout, unavailable

# 🔍 fault_type: network delay
#    ├── metric: client_error_ratio, error_ratio, rrt, timeout
#    ├── trace: latency_anomalies
#    └── log: error, exception

# 🔍 fault_type: network loss
#    ├── metric: client_error_ratio, error_ratio, rrt, timeout
#    ├── trace: latency_anomalies, request_proportion_anomalies
#    └── log: abort, disconnect, error, exception, failed, retry, timeout

# 🔍 fault_type: node cpu
#    ├── metric: node_cpu_usage_rate
#    ├── trace: request_proportion_anomalies
#    └── log: (无)

# 🔍 fault_type: node disk fill
#    ├── metric: node_filesystem_usage_rate
#    ├── trace: latency_anomalies
#    └── log: (无)

# 🔍 fault_type: node memory
#    ├── metric: node_memory_usage_rate
#    ├── trace: latency_anomalies, request_proportion_anomalies
#    └── log: error, exception, failed, refused, timeout

# 🔍 fault_type: pod failure
#    ├── metric: (无)
#    ├── trace: latency_anomalies, request_proportion_anomalies
#    └── log: error, failed, refused, timeout, unavailable

# 🔍 fault_type: target port misconfig
#    ├── metric: client_error_ratio, error_ratio
#    ├── trace: latency_anomalies, request_proportion_anomalies
#    └── log: error, exception, failed, refused, unavailable
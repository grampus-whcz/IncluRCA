# analyze_traces.py
import json
import sys
from collections import defaultdict

def analyze_trace_list_file(file_path):
    """
    分析包含多个 Trace 的 JSON 文件（格式为 JSON 数组）
    提取每个微服务（serviceName）所涉及的所有 operationName（去重）
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 注意：这里加载的是 list
    except Exception as e:
        print(f"❌ 无法读取或解析文件 {file_path}: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print("❌ 错误: 文件应为 JSON 数组，每个元素是一个 trace")
        sys.exit(1)

    # 全局汇总：service_name -> set of operationNames
    service_operations = defaultdict(set)

    # 遍历每一个 trace
    for i, trace in enumerate(data):
        print(f"🔍 正在处理第 {i+1} 个 trace (traceID: {trace.get('traceID', 'unknown')})...", end='', file=sys.stderr)

        processes_map = {}
        if 'processes' in trace:
            for proc_id, proc_info in trace['processes'].items():
                svc_name = proc_info.get('serviceName', 'unknown-service')
                processes_map[proc_id] = svc_name
        else:
            print(f"\n⚠️ 第 {i+1} 个 trace 缺少 'processes' 字段")
            continue

        if 'spans' not in trace or not isinstance(trace['spans'], list):
            print(f"\n⚠️ 第 {i+1} 个 trace 缺少或格式错误的 'spans' 字段")
            continue

        for span in trace['spans']:
            process_id = span.get('processID')
            op_name = span.get('operationName')

            if not process_id or not op_name:
                continue

            service_name = processes_map.get(process_id)
            if not service_name:
                service_name = f"unknown-service[{process_id}]"

            service_operations[service_name].add(op_name)

        print(" ✔️", file=sys.stderr)  # 处理完成标记

    return service_operations


def print_results(service_operations):
    """格式化输出结果"""
    print("\n" + "=" * 60)
    print("📊 所有微服务及其调用的操作 (operationName)")
    print("=" * 60)

    if not service_operations:
        print("🔍 未发现任何有效的 span 数据。")
        return

    for service_name in sorted(service_operations.keys()):
        operations = sorted(service_operations[service_name])
        print(f"\n🔧 微服务: {service_name}")
        for op in operations:
            print(f"   ├─ {op}")
    print("\n✅ 分析完成。\n")


def save_to_json(service_operations, output_file):
    """将结果保存为 JSON 文件"""
    result = {svc: sorted(ops) for svc, ops in service_operations.items()}
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 结果已保存到: {output_file}")
    except Exception as e:
        print(f"❌ 无法保存输出文件 {output_file}: {e}")


def main():
    if len(sys.argv) < 2:
        print(f"📌 用法: python {sys.argv[0]} <traces-json-file> [--json-output <output.json>]") 
        sys.exit(1)

    input_file = sys.argv[1]
    json_output = None

    # 解析可选参数
    args = sys.argv[1:]
    if '--json-output' in args:
        try:
            idx = args.index('--json-output')
            json_output = args[idx + 1]
        except IndexError:
            print("❌ 错误: --json-output 后需指定文件名")
            sys.exit(1)

    # 执行分析
    service_operations = analyze_trace_list_file(input_file)

    # 输出结果
    print_results(service_operations)

    # 可选：导出为 JSON
    if json_output:
        save_to_json(service_operations, json_output)


if __name__ == '__main__':
    main()
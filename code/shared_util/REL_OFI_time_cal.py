import re
from datetime import datetime

# 配置
LOG_FILE_PATH = '1.log'  # 替换为你的日志文件路径
FIRST_SAMPLE_START = "2025-09-08 22:18:07"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# 正则表达式匹配目标行
pattern = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - INFO - sample (\d+)/\d+ .*?level: (pod|service|node);"
)

def parse_log_file(log_path):
    """解析日志文件，提取时间、sample编号和类型"""
    samples = []  # 存储 (datetime_obj, sample_num, level)
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                timestamp_str = match.group(1)
                sample_num = int(match.group(2))
                level = match.group(3)
                
                try:
                    dt = datetime.strptime(timestamp_str, TIME_FORMAT)
                    samples.append((dt, sample_num, level))
                except ValueError as e:
                    print(f"时间解析错误: {timestamp_str}, 错误: {e}")
    
    # 按 sample 编号排序（确保顺序）
    samples.sort(key=lambda x: x[1])
    return samples

def calculate_durations(samples, first_start_str):
    """计算每个用例耗时、各类型总耗时、总耗时，并统计平均耗时"""
    # 插入第一个样本的开始时间
    first_dt = datetime.strptime(first_start_str, TIME_FORMAT)
    if not samples or samples[0][1] != 1:
        samples.insert(0, (first_dt, 1, None))
    else:
        if samples[0][0] != first_dt:
            samples[0] = (first_dt, 1, samples[0][2])

    durations = {}  # sample_num -> duration in seconds
    type_totals = {'pod': 0, 'service': 0, 'node': 0}
    type_counts = {'pod': 0, 'service': 0, 'node': 0}  # 新增：每类执行次数
    total_duration = 0

    # 计算每个测试用例的耗时（当前到下一个开始的时间）
    for i in range(len(samples) - 1):
        start_time = samples[i][0]
        end_time = samples[i+1][0]
        duration_sec = (end_time - start_time).total_seconds()
        
        sample_num = samples[i][1]
        level = samples[i][2]
        
        durations[sample_num] = duration_sec
        total_duration += duration_sec
        
        if level in type_totals:
            type_totals[level] += duration_sec
            type_counts[level] += 1  # 计数 +1

    # 所有用例数量
    num_samples = len(durations)
    avg_total = total_duration / num_samples if num_samples > 0 else 0

    # 计算每类平均耗时
    type_avg = {}
    for level in type_totals:
        if type_counts[level] > 0:
            type_avg[level] = type_totals[level] / type_counts[level]
        else:
            type_avg[level] = 0

    return durations, type_totals, type_counts, type_avg, total_duration, avg_total

def format_duration(seconds):
    """将秒数格式化为 HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# 主程序
if __name__ == "__main__":
    try:
        # 1. 解析日志
        samples = parse_log_file(LOG_FILE_PATH)
        
        if len(samples) < 1:
            print("未找到匹配的日志行。")
        else:
            # 2. 计算耗时与平均值
            durations, type_totals, type_counts, type_avg, total_duration, avg_total = calculate_durations(samples, FIRST_SAMPLE_START)
            
            # 3. 输出结果
            print("="*70)
            print("测试用例耗时分析")
            print("="*70)
            
            print("每个测试用例耗时 (h:min:s):")
            for sample_num, sec in sorted(durations.items()):
                fmt_time = format_duration(sec)
                print(f"  Sample {sample_num:3d}: {fmt_time}")
            
            print("\n各类型统计:")
            for level in ['pod', 'service', 'node']:
                count = type_counts[level]
                total_fmt = format_duration(type_totals[level])
                avg_fmt = format_duration(type_avg[level])
                print(f"  {level.capitalize()}: ")
                print(f"    执行次数: {count}")
                print(f"    总耗时:   {total_fmt}")
                print(f"    平均耗时: {avg_fmt}")

            print(f"\n所有测试用例统计:")
            print(f"  总用例数: {len(durations)}")
            print(f"  总耗时:   {format_duration(total_duration)}")
            print(f"  平均耗时: {format_duration(avg_total)}")

    except FileNotFoundError:
        print(f"错误：找不到日志文件 '{LOG_FILE_PATH}'，请检查路径。")
    except Exception as e:
        print(f"发生错误: {e}")
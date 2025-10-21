import pandas as pd
import glob
import json
from datetime import datetime, timedelta, timezone

from data_filter.CCF_AIOps_challenge_2025.base.base_class import BaseClass


class GroundTruthDao(BaseClass):
    def __init__(self):
        super().__init__()

    # CST 2025-06-12 00:00:00--2025-06-12 23:59:59 
    # UTC 2025-06-11 16:00:00--2025-06-12 15:59:59
    # groud_truth 全部采用北京时间
    # 数据文件名中含有的时间为 CST 时区，如 log_filebeat-server_2025-06-06_00-00-00.parquet 中的时间，表示北京时间2025年6月6号零点零分。
    # 日志数据中的时间格式为 2025-06-05T16:00:29.045Z，为 UTC 时区，表示北京时间2025年6月6号零点零分。
    # 指标数据中的时间格式为 2025-06-05T16:00:00Z ，为 UTC 时区，表示北京时间2025年6月6号零点零分。
    # 调用链数据中 startTimeMillis 的时间格式为 1749139200377，为时间戳，单位毫秒。
    def get_ground_truth(self, dataset_type: str):
        result_dict = dict()

        data_base_path = f'{self.config.data_dict["ground_truth"][dataset_type]}'

        with open(data_base_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    fault_type = data.get("fault_type")
                    instance_type = data.get("instance_type")
                    start_time_str = data.get("start_time")
                    end_time_str = data.get("end_time")
                    instance = data.get("instance")
                    if not start_time_str:
                        continue  # 跳过无 start_time 的数据

                    utc_time = datetime.strptime(start_time_str.strip('Z'), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)

                    # ✅ 如果需要转换为北京时间的可读格式
                    beijing_time = utc_time + timedelta(hours=8)

                    # 转换为字符串格式 "2025-06-06"
                    target_date_str = beijing_time.strftime("%Y-%m-%d")
                    
                    
                    # 解析为 datetime 对象（UTC）
                    start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                    end_time = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))

                    # 计算中间时间
                    mid_time = start_time + (end_time - start_time) / 2

                    # 转换为时间戳（秒，浮点数）
                    timestamp = int(mid_time.timestamp())
                    print("时间戳（秒）:", timestamp)

                    # 添加到 result_dict
                    if target_date_str not in result_dict:
                        result_dict[target_date_str] = {
                            'cloudbed' : {}
                        }
                        result_dict[target_date_str]['cloudbed']["timestamp"] = []
                        result_dict[target_date_str]['cloudbed']["failure_type"] = []
                        result_dict[target_date_str]['cloudbed']["level"] = []
                        result_dict[target_date_str]['cloudbed']["cmdb_id"] = []
                    result_dict[target_date_str]['cloudbed']["timestamp"].append(timestamp)
                    result_dict[target_date_str]['cloudbed']["failure_type"].append(fault_type)
                    result_dict[target_date_str]['cloudbed']["level"].append(instance_type)
                    
                    if type(instance) == list:
                        result_dict[target_date_str]['cloudbed']["cmdb_id"].append(instance[0])
                    else:
                        result_dict[target_date_str]['cloudbed']["cmdb_id"].append(instance)

                except Exception as e:
                    print(f"Error parsing line: {line}")
                    print(f"Exception: {e}")
                    continue

        return result_dict

    def analyze_ground_truth(self):
        def count_ground_truth_type(ground_truth_dict):
            result_dict = {
                'fault_type_count': dict(),
                'cmdb_count': dict(),
                'all_count': dict()
            }

            for date in ground_truth_dict.keys():
                for cloud_bed in ground_truth_dict[date].keys():
                    for i in range(len(ground_truth_dict[date][cloud_bed]['timestamp'])):
                        fault_type = ground_truth_dict[date][cloud_bed]['failure_type'][i]
                        cmdb_id = ground_truth_dict[date][cloud_bed]['cmdb_id'][i]
                        level = ground_truth_dict[date][cloud_bed]['level'][i]

                        if fault_type not in result_dict['fault_type_count'].keys():
                            result_dict['fault_type_count'][fault_type] = 0
                        result_dict['fault_type_count'][fault_type] += 1

                        if level == 'pod':
                            cmdb_id = f'pod/{cmdb_id.replace("-0", "").replace("-1", "").replace("-2", "")}'
                            count = 1
                        elif level == 'node':
                            cmdb_id = f'node/{cmdb_id.replace("-01", "").replace("-02", "").replace("-03", "").replace("-04", "").replace("-05", "").replace("-06", "").replace("-07", "").replace("-08", "")}'
                            count = 1
                        else:
                            cmdb_id = f'pod/{cmdb_id}'
                            count = 4
                        if cmdb_id not in result_dict['cmdb_count'].keys():
                            result_dict['cmdb_count'][cmdb_id] = 0
                        result_dict['cmdb_count'][cmdb_id] += count

                        all_info = f'{cmdb_id}/{fault_type}'
                        if all_info not in result_dict['all_count'].keys():
                            result_dict['all_count'][all_info] = 0
                        result_dict['all_count'][all_info] += count

            return result_dict

        train_valid_ground_truth = self.get_ground_truth('train_valid')
        test_ground_truth = self.get_ground_truth('test')

        train_valid_result_dict = count_ground_truth_type(train_valid_ground_truth)
        print(train_valid_result_dict)
        test_result_dict = count_ground_truth_type(test_ground_truth)
        print(test_result_dict)

    #     not_seen_fault = set(test_result_dict['all_count'].keys()) - set(train_valid_result_dict['all_count'].keys())
    #     seen_and_not_happened_fault = set(train_valid_result_dict['all_count'].keys()) - set(test_result_dict['all_count'].keys())
    #     ...
        


    def load_fault_data_by_date(self, filepath):
        """
        读取 jsonl 文件，按 start_time 判断归属日期，并组织成 result_dict。
        
        规则：
        - 如果 start_time 在当日 16:00:00 之前，归入 前一天 的 date 键下。
        - 如果 start_time 在当日 16:00:00 或之后，归入 当天 的 date 键下。
        
        Args:
            filepath (str): jsonl 文件路径
        
        Returns:
            dict: result_dict，key 为 "YYYY-MM-DD" 格式的日期字符串，value 为该日对应的数据列表。
        """
        result_dict = {}

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    uuid = data.get("uuid")
                    start_time_str = data.get("start_time")
                    if not start_time_str:
                        continue  # 跳过无 start_time 的数据

                    # 解析 ISO 8601 时间字符串
                    start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                    start_time = start_time.replace(tzinfo=None)  # 忽略时区，只处理本地时间逻辑

                    # 提取日期部分
                    date_part = start_time.date()
                    time_part = start_time.time()

                    # 判断是否在 16:00:00 之前
                    cutoff_time = datetime.strptime("16:00:00", "%H:%M:%S").time()
                    if time_part < cutoff_time:
                        # 在 16:00:00 之前，分配到 前一天
                        target_date = date_part - timedelta(days=1)
                    else:
                        # 在 16:00:00 或之后，分配到 当天
                        target_date = date_part

                    # 转换为字符串格式 "2025-06-06"
                    target_date_str = target_date.strftime("%Y-%m-%d")

                    # 添加到 result_dict
                    if target_date_str not in result_dict:
                        result_dict[target_date_str] = {
                            'cloudbed' : {}
                        }
                    result_dict[target_date_str]['cloudbed'][uuid] = {}
                    result_dict[target_date_str]['cloudbed'][uuid] = data

                except Exception as e:
                    print(f"Error parsing line: {line}")
                    print(f"Exception: {e}")
                    continue

        return result_dict


if __name__ == "__main__":
    # ground_truth_dao = GroundTruthDao()
    # filepath = "/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/phase1.jsonl"
    # result_dict = ground_truth_dao.load_fault_data_by_date(filepath)
    # print(result_dict)
    
    # # 打印结果示例
    # for date, data_list in sorted(result_dict.items()):
    #     print(f"{date}: {len(data_list)} records")



    ground_truth_dao = GroundTruthDao()
    print(ground_truth_dao.get_ground_truth('train_valid'))
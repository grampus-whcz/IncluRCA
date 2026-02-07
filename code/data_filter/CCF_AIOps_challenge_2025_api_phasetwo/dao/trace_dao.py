import sys

sys.path.append('/root/shared-nvme/work/code/RCA/IncluRCA/code')

import pandas as pd
import numpy as np
from data_filter.CCF_AIOps_challenge_2025_api.base.base_class import BaseClass
from shared_util.file_handler import FileHandler
from shared_util.time_handler import TimeHandler
import json
import copy
import os
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds


class RawTraceDao(BaseClass):
    def __init__(self):
        super().__init__()
        self.trace_pattern_dict = dict()

    def extract_trace_patterns(self):
        result_dict = dict()

        file_dict = self.config.data_dict['file']
        result_base_path = f'{self.config.param_dict["temp_data_storage"]}/analysis'

        status_code_list = []

        for dataset_type, dataset_detail_dict in file_dict.items():
            if dataset_type == 'test':
                continue
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    
                    data_base_path = f'{dataset_detail_dict["base_folder"]}/{date}/{cloud_bed}/trace-parquet/'
                    # print(data_base_path)

                    for root, dirs, files in os.walk(data_base_path):
                        index = 0
                        for file in files:                            
                            if index > 0 :
                                break
                            if file.endswith(".parquet"):
                                file_path = os.path.join(root, file)
                                print(f"Processing file: {file_path}")

                                try:
                                    # 使用 pyarrow 读取 parquet 文件
                                    table = pq.read_table(file_path)
                                    # 转换为字典列表（每行一个字典）
                                    batches = table.to_batches(max_chunksize=65536)
                                    for batch in batches:
                                        for record in batch.to_pylist():
                                            # 提取 tags 列表
                                            tags = record.get("tags", [])
                                            if not isinstance(tags, list):
                                                continue

                                            # 查找 key 为 "status.code" 的 tag
                                            for tag in tags:
                                                if isinstance(tag, dict) and (tag.get("key") == "http.status_code"                                                              
                                                                            or tag.get("key") == "grpc.status_code" 
                                                                            or tag.get("key") == "http.status_code"
                                                                            or tag.get("key") == "status.code" ):
                                                    value = tag.get("value")
                                                    # status_code_list.append(value)
                                                    if f'status_code: {value}' not in status_code_list:
                                                        status_code_list.append(f'status_code: {value}')                                                  
                                                    
                                                    break  # 每个 span 只取一次
                                except Exception as e:
                                    print(f"Error reading {file_path}: {e}")
                                    continue
                            index = index+1

        status_code_list = sorted(status_code_list)

        for service in self.config.data_dict['setting']['metric']['service_order']:
            result_dict[service] = copy.deepcopy(status_code_list)

        with open(FileHandler.set_folder(f'{result_base_path}/trace') + '/trace_patterns.json', 'w') as f:
            json.dump(result_dict, f, indent=2)

        self.logger.debug('Already extract trace patterns.')


    def extract_trace_features(self):
        file_dict = self.config.data_dict['file']
        result_base_path = f'{self.config.param_dict["temp_data_storage"]}/raw_data'

        for dataset_type, dataset_detail_dict in file_dict.items():
            result_dataset_type_path = FileHandler.set_folder(f'{result_base_path}/{dataset_type}')
            for date in dataset_detail_dict['date']:
                result_date_path = FileHandler.set_folder(f'{result_dataset_type_path}/{date}')
                timestamp_list = [TimeHandler.datetime_to_timestamp(f'{date} 00:00:00') + i * 60 for i in range(0, 24 * 60)]

                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    result_cloud_bed_path = FileHandler.set_folder(f'{result_date_path}/{cloud_bed}/raw_trace')
                    data_base_path_dir = f'{dataset_detail_dict["base_folder"]}/{date}/{cloud_bed}/trace-parquet'

                    # === 使用 pyarrow.dataset ===
                    try:
                        # 创建 dataset，自动匹配目录下所有 .parquet 文件
                        dataset = ds.dataset(data_base_path_dir, format="parquet")

                        # 获取所有文件路径（可选：用于日志）
                        file_fragments = list(dataset.get_fragments())
                        self.logger.info(f"Found {len(file_fragments)} parquet files in {data_base_path_dir}")

                        # 初始化全局聚合桶（和之前一样）
                        trace_pattern_list = [
                            f'cmdb_id: {pod}'
                            for service in self.config.data_dict['setting']['metric']['service_order']
                            for pod in RawTraceDao.rename_service2pod(service)
                        ]

                        global_temp_bucket_dict = {
                            tp: [{'parent_span': [], 'duration': [], 'span_index_dict': {}} for _ in timestamp_list]
                            for tp in trace_pattern_list
                        }
                        global_id_to_trace_pattern = [dict() for _ in timestamp_list]

                        # === 逐 batch 处理，不分文件 ===
                        scanner = dataset.scanner(batch_size=65536)  # 与 max_chunksize 对应
                        batches_iter = scanner.to_batches()

                        for batch in batches_iter:
                            self.logger.info(f"Processing batch with {batch.num_rows} rows")
                            for record in batch.to_pylist():
                                # ===== 复用你原来的 record 处理逻辑 =====
                                process = record.get("process", {})
                                if not isinstance(process, dict):
                                    continue

                                serviceName = process.get("serviceName")
                                cmdb_id = ""
                                tags = process.get("tags", [])
                                if not isinstance(tags, list):
                                    tags = []

                                if serviceName == "cartservice" or serviceName == "redis":
                                    for tag in tags:
                                        if isinstance(tag, dict) and tag.get("key") == "podName":
                                            cmdb_id = tag.get("value")
                                            break
                                else:
                                    for tag in tags:
                                        if isinstance(tag, dict) and tag.get("key") == "name":
                                            cmdb_id = tag.get("value")
                                            break

                                trace_pattern = f'cmdb_id: {cmdb_id}'
                                if cmdb_id not in self.config.data_dict['setting']['metric']['pod_order']:
                                    continue

                                start_time_sec = record.get("startTimeMillis", 0) / 1000
                                index = int((start_time_sec - timestamp_list[0]) // 60)
                                if index < 0 or index >= len(timestamp_list):
                                    continue

                                trace_id = record.get("traceID", "")
                                span_id = record.get("spanID", "")
                                parent_span = ""
                                references = record.get("references", [])
                                if references:
                                    parent_span = references[0].get("spanID", "")

                                duration = record.get("duration", 0)

                                span_key = f'{trace_id}/{span_id}'
                                parent_key = f'{trace_id}/{parent_span}'

                                if trace_pattern in global_temp_bucket_dict:
                                    bucket = global_temp_bucket_dict[trace_pattern][index]
                                    bucket['duration'].append(duration)
                                    bucket['parent_span'].append(parent_key)
                                    bucket['span_index_dict'][span_key] = len(bucket['duration']) - 1
                                    global_id_to_trace_pattern[index][span_key] = trace_pattern

                    except Exception as e:
                        self.logger.error(f"Error processing dataset {data_base_path_dir}: {e}")
                        continue

                    # === 所有数据处理完后，生成特征 ===
                    self.generate_span_features_from_aggregated_data(
                        timestamp_list, trace_pattern_list,
                        global_temp_bucket_dict, global_id_to_trace_pattern,
                        result_cloud_bed_path
                    )
                
    def generate_span_features_from_aggregated_data(self, timestamp_list, trace_pattern_list, temp_bucket_dict, id_to_trace_pattern, result_cloud_bed_path):
        span_feature_dict = {'timestamp': timestamp_list}

        # 初始化所有特征
        for trace_pattern in trace_pattern_list:
            for feature_type in ['upstream', 'current', 'downstream']:
                span_feature_dict[f'<intensity>; {trace_pattern}; type: {feature_type}'] = []
                span_feature_dict[f'<duration>; {trace_pattern}; type: {feature_type}'] = []

        # 填充 current 特征
        for i in range(len(timestamp_list)):
            for trace_pattern in trace_pattern_list:
                durations = temp_bucket_dict[trace_pattern][i]['duration']
                intensity = len(durations)
                mean_duration = np.nan_to_num(np.mean(durations)) if durations else 0.0
                span_feature_dict[f'<intensity>; {trace_pattern}; type: current'].append(intensity)
                span_feature_dict[f'<duration>; {trace_pattern}; type: current'].append(mean_duration)

        # 填充 upstream/downstream
        for i in range(len(timestamp_list)):
            child_span_dict = {'upstream': {}, 'downstream': {}}
            for trace_pattern in trace_pattern_list:
                bucket = temp_bucket_dict[trace_pattern][i]
                for idx, parent_span_id in enumerate(bucket['parent_span']):
                    duration = bucket['duration'][idx]
                    if parent_span_id not in id_to_trace_pattern[i]:
                        continue
                    parent_pattern = id_to_trace_pattern[i][parent_span_id]

                    # downstream: 当前 span 是 parent，被谁调用？
                    if parent_pattern not in child_span_dict['downstream']:
                        child_span_dict['downstream'][parent_pattern] = {'intensity': 0, 'duration': []}
                    child_span_dict['downstream'][parent_pattern]['intensity'] += 1
                    child_span_dict['downstream'][parent_pattern]['duration'].append(duration)

                    # upstream: 当前 span 的 parent 是谁？
                    if parent_span_id in temp_bucket_dict[parent_pattern][i]['span_index_dict']:
                        if trace_pattern not in child_span_dict['upstream']:
                            child_span_dict['upstream'][trace_pattern] = {'intensity': 0, 'duration': []}
                        child_span_dict['upstream'][trace_pattern]['intensity'] += 1
                        child_span_dict['upstream'][trace_pattern]['duration'].append(duration)

            # 写入特征
            for trace_pattern in trace_pattern_list:
                for feature_type in ['upstream', 'downstream']:
                    if trace_pattern in child_span_dict[feature_type]:
                        val = child_span_dict[feature_type][trace_pattern]
                        span_feature_dict[f'<intensity>; {trace_pattern}; type: {feature_type}'].append(val['intensity'])
                        mean_dur = np.nan_to_num(np.mean(val['duration'])) if val['duration'] else 0.0
                        span_feature_dict[f'<duration>; {trace_pattern}; type: {feature_type}'].append(mean_dur)
                    else:
                        span_feature_dict[f'<intensity>; {trace_pattern}; type: {feature_type}'].append(0)
                        span_feature_dict[f'<duration>; {trace_pattern}; type: {feature_type}'].append(0.0)

        # === 最终写入一次 ===
        span_feature_df = pd.DataFrame(span_feature_dict)
        output_path = f'{result_cloud_bed_path}/span_features.csv'
        span_feature_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved aggregated span features to {output_path}")

    # def extract_trace_features(self):
    #     file_dict = self.config.data_dict['file']
    #     result_base_path = f'{self.config.param_dict["temp_data_storage"]}/raw_data'

    #     for dataset_type, dataset_detail_dict in file_dict.items():
    #         result_dataset_type_path = FileHandler.set_folder(f'{result_base_path}/{dataset_type}')
    #         for date in dataset_detail_dict['date']:
    #             result_date_path = FileHandler.set_folder(f'{result_dataset_type_path}/{date}')
    #             timestamp_list = [TimeHandler.datetime_to_timestamp(f'{date} 00:00:00') + i * 60 for i in range(0, 24 * 60)]

    #             for cloud_bed in dataset_detail_dict['cloud_bed']:
                    
    #                 result_cloud_bed_path = FileHandler.set_folder(f'{result_date_path}/{cloud_bed}/raw_trace')
    #                 data_base_path_dir = f'{dataset_detail_dict["base_folder"]}/{date}/{cloud_bed}/trace-parquet'
                    
    #                 for root, dirs, files in os.walk(data_base_path_dir):
    #                     sorted_files = sorted(files)
    #                     for file in sorted_files:
                            
    #                         if file.endswith(".parquet"):
    #                             data_base_path = os.path.join(root, file)

    #                             self.logger.info(f'Preprocessing traces: dataset_type: {dataset_type}, date: {date}, '
    #                                              f'cloudbed: {cloud_bed},'
    #                                              f'path: {data_base_path}.')
    #                             self.process_traces(timestamp_list, data_base_path, result_cloud_bed_path)


    # def process_traces(self, timestamp_list: list, data_base_path: str, result_base_path: str):
    #     trace_pattern_list = []
    #     for service in self.config.data_dict['setting']['metric']['service_order']:
    #         pod_list = RawTraceDao.rename_service2pod(service)
    #         for pod in pod_list:
    #             trace_pattern_list.append(f'cmdb_id: {pod}')

    #     temp_bucket_dict = dict()
    #     for i in trace_pattern_list:
    #         temp_bucket_dict[i] = [{'parent_span': [],
    #                                 'duration': [],
    #                                 'span_index_dict': dict()} for _ in timestamp_list]

    #     id_to_trace_pattern = [dict() for _ in timestamp_list]
        
        
    #     table = pq.read_table(data_base_path)
    #     print(data_base_path)
    #     batches = table.to_batches(max_chunksize=65536)
    #     for batch in batches:
    #         for record in batch.to_pylist():
    #             # 提取 tags 列表
    #             process = record.get("process", [])
    #             if not isinstance(process, dict):
    #                 continue

    #             serviceName = process.get("serviceName")
    #             # 查找 key 为 "status.code" 的 tag
    #             cmdb_id = ""
    #             if serviceName == "cartservice" or serviceName == "redis":
    #                 for tag in process.get("tags"):
    #                     if isinstance(tag, dict) and tag.get("key") == "podName":
    #                         cmdb_id = tag.get("value")
    #                         break
    #             else:                    
    #                 for tag in process.get("tags"):
    #                     if isinstance(tag, dict) and tag.get("key") == "name":
    #                         cmdb_id = tag.get("value")
    #                         break
    #             trace_pattern = f'cmdb_id: {cmdb_id}'
    #             if cmdb_id not in self.config.data_dict['setting']['metric']['pod_order']:
    #                 continue


    #             index = int((record.get("startTimeMillis") / 1000 - timestamp_list[0]) / 60)
    #             # print(record.get("startTimeMillis"))

    #             trace_id = record.get("traceID")
    #             span_id = record.get("spanID")
    #             parent_span = ""
    #             if len(record.get("references")) > 0:
    #                 parent_span = record.get("references")[0].get("spanID")
                    
    #             duration = record.get("duration")
    #             id_to_trace_pattern[index][f'{trace_id}/{span_id}'] = f'cmdb_id: {cmdb_id}'
    #             temp_bucket_dict[trace_pattern][index]['span_index_dict'][f'{trace_id}/{span_id}'] = len(temp_bucket_dict[trace_pattern][index]['duration'])
    #             temp_bucket_dict[trace_pattern][index]['parent_span'].append(f'{trace_id}/{parent_span}')
    #             temp_bucket_dict[trace_pattern][index]['duration'].append(duration)    

    #     span_feature_dict = {'timestamp': timestamp_list}

    #     for trace_pattern in trace_pattern_list:
    #         span_feature_dict[f'<intensity>; {trace_pattern}; type: upstream'] = []
    #         span_feature_dict[f'<duration>; {trace_pattern}; type: upstream'] = []
    #         span_feature_dict[f'<intensity>; {trace_pattern}; type: current'] = []
    #         span_feature_dict[f'<duration>; {trace_pattern}; type: current'] = []
    #         span_feature_dict[f'<intensity>; {trace_pattern}; type: downstream'] = []
    #         span_feature_dict[f'<duration>; {trace_pattern}; type: downstream'] = []

    #         for i in range(len(timestamp_list)):
    #             span_feature_dict[f'<intensity>; {trace_pattern}; type: current'].append(len(temp_bucket_dict[trace_pattern][i]['duration']))
    #             if len(temp_bucket_dict[trace_pattern][i]['duration']) > 0:
    #                 span_feature_dict[f'<duration>; {trace_pattern}; type: current'].append(np.nan_to_num(np.nanmean(temp_bucket_dict[trace_pattern][i]['duration'])))
    #             else:
    #                 span_feature_dict[f'<duration>; {trace_pattern}; type: current'].append(0.0)

    #     for i in range(len(timestamp_list)):
    #         child_span_dict = {
    #             'upstream': dict(),
    #             'downstream': dict()
    #         }
    #         for trace_pattern in trace_pattern_list:
    #             for index in range(len(temp_bucket_dict[trace_pattern][i]['parent_span'])):
    #                 parent_span_id = temp_bucket_dict[trace_pattern][i]['parent_span'][index]
    #                 if parent_span_id not in id_to_trace_pattern[i].keys():
    #                     continue
    #                 parent_span_pattern = id_to_trace_pattern[i][parent_span_id]
    #                 if parent_span_pattern not in child_span_dict['downstream'].keys():
    #                     child_span_dict['downstream'][parent_span_pattern] = {
    #                         'intensity': 0,
    #                         'duration': []
    #                     }
    #                 child_span_dict['downstream'][parent_span_pattern]['intensity'] += 1
    #                 child_span_dict['downstream'][parent_span_pattern]['duration'].append(
    #                     temp_bucket_dict[trace_pattern][i]['duration'][index])

    #                 if parent_span_id not in temp_bucket_dict[parent_span_pattern][i]['span_index_dict'].keys():
    #                     continue

    #                 temp_index = temp_bucket_dict[parent_span_pattern][i]['span_index_dict'][parent_span_id]
    #                 if trace_pattern not in child_span_dict['upstream'].keys():
    #                     child_span_dict['upstream'][trace_pattern] = {
    #                         'intensity': 0,
    #                         'duration': []
    #                     }
    #                 child_span_dict['upstream'][trace_pattern]['intensity'] += 1
    #                 child_span_dict['upstream'][trace_pattern]['duration'].append(temp_bucket_dict[parent_span_pattern][i]['duration'][temp_index])

    #         for trace_pattern in trace_pattern_list:
    #             for feature_type in ['upstream', 'downstream']:
    #                 if trace_pattern in child_span_dict[feature_type].keys():
    #                     value_dict = child_span_dict[feature_type][trace_pattern]
    #                     span_feature_dict[f'<intensity>; {trace_pattern}; type: {feature_type}'].append(value_dict['intensity'])
    #                     if len(value_dict['duration']) > 0:
    #                         span_feature_dict[f'<duration>; {trace_pattern}; type: {feature_type}'].append(np.nan_to_num(np.nanmean(value_dict['duration'])))
    #                     else:
    #                         span_feature_dict[f'<duration>; {trace_pattern}; type: {feature_type}'].append(0.0)
    #                 else:
    #                     span_feature_dict[f'<intensity>; {trace_pattern}; type: {feature_type}'].append(0)
    #                     span_feature_dict[f'<duration>; {trace_pattern}; type: {feature_type}'].append(0.0)

    #     span_feature_df = pd.DataFrame(span_feature_dict)

    #     span_feature_df.to_csv(f'{result_base_path}/span_features.csv', index=False)

    def load_trace_csv(self):
        result_dict = dict()

        file_dict = self.config.data_dict['file']
        trace_base_path = f'{self.config.param_dict["temp_data_storage"]}/raw_data'

        for dataset_type, dataset_detail_dict in file_dict.items():
            result_dict[dataset_type] = dict()
            trace_dataset_type_path = f'{trace_base_path}/{dataset_type}'
            for date in dataset_detail_dict['date']:
                result_dict[dataset_type][date] = dict()
                trace_date_path = f'{trace_dataset_type_path}/{date}'
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    
                    result_dict[dataset_type][date][cloud_bed] = dict()
                    trace_cloud_bed_path = f'{trace_date_path}/{cloud_bed}/raw_trace'

                    result_dict[dataset_type][date][cloud_bed]['span_features'] = pd.read_csv(f'{trace_cloud_bed_path}/span_features.csv')

        return result_dict
    

def extract_grpc_status_codes_from_parquet_dir(parquet_dir):
    """
    遍历目录下的所有 .parquet 文件，提取每个 span 中 rpc.grpc.status_code 的值。

    Args:
        parquet_dir (str): 包含 trace parquet 文件的目录路径

    Returns:
        list: 所有提取到的 rpc.grpc.status_code 值（仅 value，如 'OK', 1, 'Unknown' 等）
    """
    status_code_list = []

    # 遍历目录下所有文件
    for root, dirs, files in os.walk(parquet_dir):
        index = 0
        for file in files:
            if index > 0 :
                break
            if file.endswith(".parquet"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                try:
                    # 使用 pyarrow 读取 parquet 文件
                    table = pq.read_table(file_path)
                    # 转换为字典列表（每行一个字典）
                    batches = table.to_batches(max_chunksize=65536)
                    for batch in batches:
                        for record in batch.to_pylist():
                            # 提取 tags 列表
                            tags = record.get("tags", [])
                            if not isinstance(tags, list):
                                continue

                            # 查找 key 为 "status.code" 的 tag
                            for tag in tags:
                                if isinstance(tag, dict) and (tag.get("key") == "http.status_code"                                                              
                                                              or tag.get("key") == "grpc.status_code" 
                                                              or tag.get("key") == "http.status_code"
                                                              or tag.get("key") == "status.code" ):
                                    value = tag.get("value")
                                    status_code_list.append(value)
                                    break  # 每个 span 只取一次
                                print(tag)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
            index = index+1

    return status_code_list


def get_parquet_row_count(file_path):
    try:
        # 读取 Parquet 文件的 metadata
        parquet_file = pq.ParquetFile(file_path)
        row_count = parquet_file.metadata.num_rows
        return row_count
    except Exception as e:
        print(f"Error reading file: {e}")
        return None




# 使用示例
# if __name__ == "__main__":
    
#     # 使用示例
#     file_path = "/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/2025-06-06/cloudbed/trace-parquet/trace_jaeger-span_2025-06-06_00-00-00.parquet"
#     row_count = get_parquet_row_count(file_path)

#     if row_count is not None:
#         print(f"Total number of rows in '{file_path}': {row_count:,}")
    
#     parquet_dir = "/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/2025-06-06/cloudbed/trace-parquet"
#     status_code_list = extract_grpc_status_codes_from_parquet_dir(parquet_dir)

#     # 打印结果统计
#     print(f"Total rpc.grpc.status_code values found: {len(status_code_list)}")
#     from collections import Counter
#     print("Top status codes:", Counter(status_code_list).most_common())


if __name__ == '__main__':
    raw_trace_dao = RawTraceDao()
    # raw_trace_dao.extract_trace_patterns()

    raw_trace_dao.extract_trace_features()

    # feature_dict = raw_trace_dao.load_trace_csv()


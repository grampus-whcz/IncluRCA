import sys

sys.path.append('/root/shared-nvme/work/code/Repdf/code')

import pandas as pd
import numpy as np
from data_filter.CCF_AIOps_challenge_2022.base.base_class import BaseClass
from shared_util.file_handler import FileHandler
from shared_util.time_handler import TimeHandler
import json
import copy


class RawApiDao(BaseClass):
    def __init__(self):
        super().__init__()
        self.trace_pattern_dict = dict()
        
    def get_pod_name(self, trace_pattern):
        prefix = 'cmdb_id: '
        if trace_pattern.startswith(prefix):
            return trace_pattern[len(prefix):]
        return None  # 或者抛出异常，根据需要处理非法格式

    def extract_api_features(self):
        file_dict = self.config.data_dict['file']
        result_base_path = f'{self.config.param_dict["temp_data_storage"]}/raw_data'

        for dataset_type, dataset_detail_dict in file_dict.items():
            result_dataset_type_path = FileHandler.set_folder(f'{result_base_path}/{dataset_type}')
            for date in dataset_detail_dict['date']:
                result_date_path = FileHandler.set_folder(f'{result_dataset_type_path}/{date}')
                timestamp_list = [TimeHandler.datetime_to_timestamp(f'{date} 00:00:00') + i * 60 for i in range(0, 24 * 60)]

                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue

                    result_cloud_bed_path = FileHandler.set_folder(f'{result_date_path}/{cloud_bed}/raw_trace')
                    data_base_path = f'{dataset_detail_dict["base_folder"]}/{date}/{cloud_bed}/trace/all/trace_jaeger-span.csv'

                    self.logger.debug(f'Preprocessing traces: dataset_type: {dataset_type}, date: {date}, '
                                      f'cloudbed: {cloud_bed}.')
                    self.process_traces(timestamp_list, data_base_path, result_cloud_bed_path)

    def process_traces(self, timestamp_list: list, data_base_path: str, result_base_path: str):
        trace_pattern_list = []
        for service in self.config.data_dict['setting']['metric']['service_order']:
            pod_list = RawApiDao.rename_service2pod(service)
            for pod in pod_list:
                trace_pattern_list.append(f'cmdb_id: {pod}')
                
        api_pattern_list = []
        for api in self.config.data_dict['setting']['metric']['api_order']:
            api_pattern_list.append(f'{api}')

        temp_bucket_dict = dict()
        for i in api_pattern_list:
            temp_bucket_dict[i] = [{'parent_span': [],
                                    'duration': [],
                                    'span_index_dict': dict()} for _ in timestamp_list]

        id_to_api_pattern = [dict() for _ in timestamp_list]

        reader = pd.read_csv(data_base_path, chunksize=100000)
        print("processing: ", data_base_path)
        chunk_num = 0
        for chunk in reader:
            chunk_num += 1
            print(f"Processing chunk {chunk_num}")
            # if chunk_num > 12:
            #     break
            trace_df = chunk
            for _, row in trace_df.iterrows():
                api_pattern = row["operation_name"]
                if row["cmdb_id"] not in self.config.data_dict['setting']['metric']['pod_order']:
                    continue

                index = int((row["timestamp"] / 1000 - timestamp_list[0]) / 60)
                if api_pattern not in self.config.data_dict['setting']['metric']['api_order']:
                    continue

                id_to_api_pattern[index][f'{row["trace_id"]}/{row["span_id"]}'] = f'{row["operation_name"]}'
                temp_bucket_dict[api_pattern][index]['span_index_dict'][f'{row["trace_id"]}/{row["span_id"]}'] = len(temp_bucket_dict[api_pattern][index]['duration'])
                temp_bucket_dict[api_pattern][index]['parent_span'].append(f'{row["trace_id"]}/{row["parent_span"]}')
                temp_bucket_dict[api_pattern][index]['duration'].append(row["duration"])

        span_feature_dict = {'timestamp': timestamp_list}

        for api_pattern in api_pattern_list:
            span_feature_dict[f'<intensity>; {api_pattern}; type: upstream'] = []
            span_feature_dict[f'<duration>; {api_pattern}; type: upstream'] = []
            span_feature_dict[f'<intensity>; {api_pattern}; type: current'] = []
            span_feature_dict[f'<duration>; {api_pattern}; type: current'] = []
            span_feature_dict[f'<intensity>; {api_pattern}; type: downstream'] = []
            span_feature_dict[f'<duration>; {api_pattern}; type: downstream'] = []

            for i in range(len(timestamp_list)):
                span_feature_dict[f'<intensity>; {api_pattern}; type: current'].append(len(temp_bucket_dict[api_pattern][i]['duration']))
                if len(temp_bucket_dict[api_pattern][i]['duration']) > 0:
                    span_feature_dict[f'<duration>; {api_pattern}; type: current'].append(np.nan_to_num(np.nanmean(temp_bucket_dict[api_pattern][i]['duration'])))
                else:
                    span_feature_dict[f'<duration>; {api_pattern}; type: current'].append(0.0)

        for i in range(len(timestamp_list)):
            child_span_dict = {
                'upstream': dict(),
                'downstream': dict()
            }
            for api_pattern in api_pattern_list:
                for index in range(len(temp_bucket_dict[api_pattern][i]['parent_span'])):
                    parent_span_id = temp_bucket_dict[api_pattern][i]['parent_span'][index]
                    if parent_span_id not in id_to_api_pattern[i].keys():
                        continue
                    parent_span_pattern = id_to_api_pattern[i][parent_span_id]
                    if parent_span_pattern not in child_span_dict['downstream'].keys():
                        child_span_dict['downstream'][parent_span_pattern] = {
                            'intensity': 0,
                            'duration': []
                        }
                    child_span_dict['downstream'][parent_span_pattern]['intensity'] += 1
                    child_span_dict['downstream'][parent_span_pattern]['duration'].append(
                        temp_bucket_dict[api_pattern][i]['duration'][index])
                        
                    # parent_span_api = temp_bucket_dict[api_pattern][i]['parent_span_api'][index]

                    # if parent_span_api == "NA":
                    #     continue
                    
                    if parent_span_id not in temp_bucket_dict[parent_span_pattern][i]['span_index_dict'].keys():
                        continue

                    temp_index = temp_bucket_dict[parent_span_pattern][i]['span_index_dict'][parent_span_id]
                    if api_pattern not in child_span_dict['upstream'].keys():
                        child_span_dict['upstream'][api_pattern] = {
                            'intensity': 0,
                            'duration': []
                        }
                    child_span_dict['upstream'][api_pattern]['intensity'] += 1
                    child_span_dict['upstream'][api_pattern]['duration'].append(temp_bucket_dict[parent_span_pattern][i]['duration'][temp_index])

            for api_pattern in api_pattern_list:
                for feature_type in ['upstream', 'downstream']:
                    if api_pattern in child_span_dict[feature_type].keys():
                        value_dict = child_span_dict[feature_type][api_pattern]
                        span_feature_dict[f'<intensity>; {api_pattern}; type: {feature_type}'].append(value_dict['intensity'])
                        if len(value_dict['duration']) > 0:
                            span_feature_dict[f'<duration>; {api_pattern}; type: {feature_type}'].append(np.nan_to_num(np.nanmean(value_dict['duration'])))
                        else:
                            span_feature_dict[f'<duration>; {api_pattern}; type: {feature_type}'].append(0.0)
                    else:
                        span_feature_dict[f'<intensity>; {api_pattern}; type: {feature_type}'].append(0)
                        span_feature_dict[f'<duration>; {api_pattern}; type: {feature_type}'].append(0.0)

        span_feature_df = pd.DataFrame(span_feature_dict)
        print("writing to csv: ", f'{result_base_path}/span_api_features.csv')
        span_feature_df.to_csv(f'{result_base_path}/span_api_features.csv', index=False)

    def load_api_csv(self):
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
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    result_dict[dataset_type][date][cloud_bed] = dict()
                    trace_cloud_bed_path = f'{trace_date_path}/{cloud_bed}/raw_trace'

                    result_dict[dataset_type][date][cloud_bed]['span_api_features'] = pd.read_csv(f'{trace_cloud_bed_path}/span_api_features.csv')

        return result_dict


if __name__ == '__main__':
    raw_api_dao = RawApiDao()

    raw_api_dao.extract_api_features()

    feature_dict = raw_api_dao.load_api_csv()


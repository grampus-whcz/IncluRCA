import os
import sys

sys.path.append('/root/shared-nvme/work/code/RCA/LasRCA/code')

import pandas as pd
import numpy as np
from data_filter.CCF_AIOps_challenge_2022.base.base_class import BaseClass
from shared_util.time_handler import TimeHandler
from shared_util.common import *
import json
import copy

# cd /root/shared-nvme/work/code/RCA/LasRCA/code
# nohup python -W ignore -u ./data_filter/CCF_AIOps_challenge_2022/dao/trace_dao.py > ./data_filter/CCF_AIOps_challenge_2022/dao/trace_dao.log 2>&1 &


class RawTraceDao(BaseClass):
    def __init__(self):
        super().__init__()
        self.trace_pattern_dict = dict()
        
    def get_pod_name(self, trace_pattern):
        prefix = 'cmdb_id: '
        if trace_pattern.startswith(prefix):
            return trace_pattern[len(prefix):]
        return None  # 或者抛出异常，根据需要处理非法格式

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
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    data_base_path = f'{dataset_detail_dict["base_folder"]}/{date}/{cloud_bed}/trace/all/trace_jaeger-span.csv'
                    reader = pd.read_csv(data_base_path, chunksize=100000)
                    for chunk in reader:
                        trace_df = chunk
                        for _, row in trace_df.iterrows():
                            if f'status_code: {row["status_code"]}' not in status_code_list:
                                status_code_list.append(f'status_code: {row["status_code"]}')
        status_code_list = sorted(status_code_list)

        for service in self.config.data_dict['setting']['metric']['service_order']:
            result_dict[service] = copy.deepcopy(status_code_list)

        os.makedirs(f'{result_base_path}/trace')
        with open(f'{result_base_path}/trace/trace_patterns.json', 'w') as f:
            json.dump(result_dict, f, indent=2)

        logger.debug('Already extract trace patterns.')

    def extract_trace_features(self):
        file_dict = self.config.data_dict['file']
        result_base_path = f'{self.config.param_dict["temp_data_storage"]}/raw_data'

        for dataset_type, dataset_detail_dict in file_dict.items():
            result_dataset_type_path = f'{result_base_path}/{dataset_type}'
            os.makedirs(result_dataset_type_path, exist_ok=True)
            for date in dataset_detail_dict['date']:
                result_date_path = f'{result_dataset_type_path}/{date}'
                os.makedirs(result_date_path, exist_ok=True)
                timestamp_list = [TimeHandler.datetime_to_timestamp(f'{date} 00:00:00') + i * 60 for i in range(0, 24 * 60)]

                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue

                    result_cloud_bed_path = f'{result_date_path}/{cloud_bed}/raw_trace'
                    os.makedirs(result_cloud_bed_path, exist_ok=True)
                    data_base_path = f'{dataset_detail_dict["base_folder"]}/{date}/{cloud_bed}/trace/all/trace_jaeger-span.csv'

                    logger.debug(f'Preprocessing traces: dataset_type: {dataset_type}, date: {date}, '
                                      f'cloudbed: {cloud_bed}.')
                    self.process_traces(timestamp_list, data_base_path, result_cloud_bed_path)

    def process_traces(self, timestamp_list: list, data_base_path: str, result_base_path: str):
        trace_pattern_list = []
        for service in self.config.data_dict['setting']['metric']['service_order']:
            pod_list = RawTraceDao.rename_service2pod(service)
            for pod in pod_list:
                trace_pattern_list.append(f'cmdb_id: {pod}')
        
        # added for api     
        api_pattern_list = []
        for api in self.config.data_dict['setting']['metric']['api_order']:
            api_pattern_list.append(f'{api}')

        temp_bucket_dict = dict()
        for i in trace_pattern_list:
            temp_bucket_dict[i] = [{'parent_span_pattern': [],  # new added feature, about the parent span pattern which is the parent span of the current span
                                    'parent_span': [],          # just the parent span id
                                    'duration': [],
                                    'span_index_dict': dict()} for _ in timestamp_list]
          
        # added for api  
        api_temp_bucket_dict = dict()
        for i in api_pattern_list:
            api_temp_bucket_dict[i] = dict()
            for j in self.config.data_dict['setting']['metric']['api_dict'][i]:
                api_temp_bucket_dict[i][j] = [{'parent_span_pattern': [],
                                               'parent_span_api':[],
                                             'parent_span': [],
                                             'duration': [],
                                             'span_index_dict': dict()} for _ in timestamp_list]


        id_to_trace_pattern = [dict() for _ in timestamp_list]
        
        id_to_api_pattern = [dict() for _ in timestamp_list]

        reader = pd.read_csv(data_base_path, chunksize=100000)
        chunk_num = 0
        for chunk in reader:
            # for the data is very large, just debug the first two chunks
            chunk_num += 1
            print(f"Processing chunk {chunk_num}")
            # if chunk_num > 12:
            #     break
            # debug end

            trace_df = chunk
            for _, row in trace_df.iterrows():
                trace_pattern = f'cmdb_id: {row["cmdb_id"]}'
                if row["cmdb_id"] not in self.config.data_dict['setting']['metric']['pod_order']:
                    continue

                index = int((row["timestamp"] / 1000 - timestamp_list[0]) / 60)

                id_to_trace_pattern[index][f'{row["trace_id"]}/{row["span_id"]}'] = f'cmdb_id: {row["cmdb_id"]}'
                temp_bucket_dict[trace_pattern][index]['span_index_dict'][f'{row["trace_id"]}/{row["span_id"]}'] = len(temp_bucket_dict[trace_pattern][index]['duration'])
                temp_bucket_dict[trace_pattern][index]['parent_span'].append(f'{row["trace_id"]}/{row["parent_span"]}')
                temp_bucket_dict[trace_pattern][index]['duration'].append(row["duration"])
                
                # added for api
                api = row["operation_name"]
                if api not in self.config.data_dict['setting']['metric']['api_order']:
                    continue
                id_to_api_pattern[index][f'{row["trace_id"]}/{row["span_id"]}'] = f'{row["operation_name"]}'
                api_temp_bucket_dict[api][f'{row["cmdb_id"]}'][index]['span_index_dict'][f'{row["trace_id"]}/{row["span_id"]}'] = len(api_temp_bucket_dict[api][f'{row["cmdb_id"]}'][index]['duration'])
                api_temp_bucket_dict[api][f'{row["cmdb_id"]}'][index]['parent_span'].append(f'{row["trace_id"]}/{row["parent_span"]}')
                api_temp_bucket_dict[api][f'{row["cmdb_id"]}'][index]['duration'].append(row["duration"])
                

        reader = pd.read_csv(data_base_path, chunksize=100000)
        chunk_num = 0
        for chunk in reader:
            # for the data is very large, just debug the first two chunks
            chunk_num += 1
            print(f"Processing chunk {chunk_num}")
            # if chunk_num > 12:
            #     break
            # debug end

            trace_df = chunk
            for _, row in trace_df.iterrows():
                trace_pattern = f'cmdb_id: {row["cmdb_id"]}'
                if row["cmdb_id"] not in self.config.data_dict['setting']['metric']['pod_order']:
                    continue

                index = int((row["timestamp"] / 1000 - timestamp_list[0]) / 60)
                parent_span_id = f'{row["trace_id"]}/{row["parent_span"]}'
                
                # added for api
                api = row["operation_name"]
                if api not in self.config.data_dict['setting']['metric']['api_order']:
                    continue
                
                flag = False
                if index == 0:
                    if parent_span_id in id_to_trace_pattern[index+1].keys():
                        temp_bucket_dict[trace_pattern][index]['parent_span_pattern'].append(id_to_trace_pattern[index+1][f'{row["trace_id"]}/{row["parent_span"]}'])
                        flag = True
                    elif parent_span_id in id_to_trace_pattern[index].keys():
                        temp_bucket_dict[trace_pattern][index]['parent_span_pattern'].append(id_to_trace_pattern[index][f'{row["trace_id"]}/{row["parent_span"]}'])
                        flag = True
                    else:
                        pass
                elif index == 1439:
                    if parent_span_id in id_to_trace_pattern[index-1].keys():
                        temp_bucket_dict[trace_pattern][index]['parent_span_pattern'].append(id_to_trace_pattern[index-1][f'{row["trace_id"]}/{row["parent_span"]}'])
                        flag = True
                    elif parent_span_id in id_to_trace_pattern[index].keys():
                        temp_bucket_dict[trace_pattern][index]['parent_span_pattern'].append(id_to_trace_pattern[index][f'{row["trace_id"]}/{row["parent_span"]}'])
                        flag = True
                    else:
                        pass
                else:
                    if parent_span_id in id_to_api_pattern[index-1].keys():
                        temp_bucket_dict[trace_pattern][index]['parent_span_pattern'].append(id_to_trace_pattern[index-1][f'{row["trace_id"]}/{row["parent_span"]}'])
                        flag = True                    
                    elif parent_span_id in id_to_api_pattern[index+1].keys():
                        temp_bucket_dict[trace_pattern][index]['parent_span_pattern'].append(id_to_trace_pattern[index+1][f'{row["trace_id"]}/{row["parent_span"]}'])
                        flag = True
                    elif parent_span_id in id_to_api_pattern[index].keys():
                        temp_bucket_dict[trace_pattern][index]['parent_span_pattern'].append(id_to_trace_pattern[index][f'{row["trace_id"]}/{row["parent_span"]}'])
                        flag = True
                    else:
                        pass
                if not flag:
                    temp_bucket_dict[trace_pattern][index]['parent_span_pattern'].append("NA")
                    
                flag_api = False
                if index == 0:
                    if parent_span_id in id_to_api_pattern[index+1].keys():
                        api_temp_bucket_dict[api][f'{row["cmdb_id"]}'][index]['parent_span_api'].append(id_to_api_pattern[index+1][f'{row["trace_id"]}/{row["parent_span"]}'])
                        flag_api = True
                    elif parent_span_id in id_to_api_pattern[index].keys():
                        api_temp_bucket_dict[api][f'{row["cmdb_id"]}'][index]['parent_span_api'].append(id_to_api_pattern[index][f'{row["trace_id"]}/{row["parent_span"]}'])
                        flag_api = True
                    else:
                        pass
                elif index == 1439:
                    if parent_span_id in id_to_api_pattern[index-1].keys():
                        api_temp_bucket_dict[api][f'{row["cmdb_id"]}'][index]['parent_span_api'].append(id_to_api_pattern[index-1][f'{row["trace_id"]}/{row["parent_span"]}'])
                        flag_api = True
                    elif parent_span_id in id_to_api_pattern[index].keys():
                        api_temp_bucket_dict[api][f'{row["cmdb_id"]}'][index]['parent_span_api'].append(id_to_api_pattern[index][f'{row["trace_id"]}/{row["parent_span"]}'])
                        flag_api = True
                    else:
                        pass
                else:
                    if parent_span_id in id_to_api_pattern[index-1].keys():
                        api_temp_bucket_dict[api][f'{row["cmdb_id"]}'][index]['parent_span_api'].append(id_to_api_pattern[index-1][f'{row["trace_id"]}/{row["parent_span"]}'])
                        flag_api = True                    
                    elif parent_span_id in id_to_api_pattern[index+1].keys():
                        api_temp_bucket_dict[api][f'{row["cmdb_id"]}'][index]['parent_span_api'].append(id_to_api_pattern[index+1][f'{row["trace_id"]}/{row["parent_span"]}'])
                        flag_api = True
                    elif parent_span_id in id_to_api_pattern[index].keys():
                        api_temp_bucket_dict[api][f'{row["cmdb_id"]}'][index]['parent_span_api'].append(id_to_api_pattern[index][f'{row["trace_id"]}/{row["parent_span"]}'])
                        flag_api = True
                    else:
                        pass
                if not flag_api:
                    api_temp_bucket_dict[api][f'{row["cmdb_id"]}'][index]['parent_span_api'].append("NA")
                        
        with open(f'{result_base_path}/temp_bucket_dict.pkl', 'wb') as f:
            pickle.dump(temp_bucket_dict, f)

        # with open(f'{result_base_path}/api_temp_bucket_dict.pkl', 'wb') as f:
        #     pickle.dump(api_temp_bucket_dict, f)

        # added for api ####################
        # temp_bucket_dict should contains api patterns duration
        api_feature_dict = {'timestamp': timestamp_list}        
                
        for api_pattern in api_pattern_list:
            for api_pod in self.config.data_dict['setting']['metric']['api_dict'][api_pattern]:
                api_feature_dict[f'<intensity>; {api_pattern}; {api_pod}; type: upstream'] = []
                api_feature_dict[f'<duration>; {api_pattern}; {api_pod}; type: upstream'] = []
                api_feature_dict[f'<intensity>; {api_pattern}; {api_pod}; type: current'] = []
                api_feature_dict[f'<duration>; {api_pattern}; {api_pod}; type: current'] = []
                api_feature_dict[f'<intensity>; {api_pattern}; {api_pod}; type: downstream'] = []
                api_feature_dict[f'<duration>; {api_pattern}; {api_pod}; type: downstream'] = []

                for i in range(len(timestamp_list)):
                    api_feature_dict[f'<intensity>; {api_pattern}; {api_pod}; type: current'].append(len(api_temp_bucket_dict[api_pattern][api_pod][i]['duration']))
                    if len(api_temp_bucket_dict[api_pattern][api_pod][i]['duration']) > 0:
                        api_feature_dict[f'<duration>; {api_pattern}; {api_pod}; type: current'].append(np.nan_to_num(np.nanmean(api_temp_bucket_dict[api_pattern][api_pod][i]['duration'])))
                    else:
                        api_feature_dict[f'<duration>; {api_pattern}; {api_pod}; type: current'].append(0.0)

        for i in range(len(timestamp_list)):
            for api_pattern in api_pattern_list:
                child_span_dict = {
                    'upstream': dict(),
                    'downstream': dict()
                }
                for api_pod in self.config.data_dict['setting']['metric']['api_dict'][api_pattern]:
                    for index in range(len(api_temp_bucket_dict[api_pattern][api_pod][i]['parent_span'])):
                        parent_span_id = api_temp_bucket_dict[api_pattern][api_pod][i]['parent_span'][index]
                        if parent_span_id not in id_to_trace_pattern[i].keys():
                            continue
                        parent_span_pattern = id_to_trace_pattern[i][parent_span_id]
                        if parent_span_pattern not in child_span_dict['downstream'].keys():
                            child_span_dict['downstream'][parent_span_pattern] = {
                                'intensity': 0,
                                'duration': []
                            }
                        child_span_dict['downstream'][parent_span_pattern]['intensity'] += 1
                        child_span_dict['downstream'][parent_span_pattern]['duration'].append(
                            api_temp_bucket_dict[api_pattern][api_pod][i]['duration'][index])
                        
                        parent_span = self.get_pod_name(parent_span_pattern)
                        
                        parent_span_api = api_temp_bucket_dict[api_pattern][api_pod][i]['parent_span_api'][index]

                        if parent_span_api == "NA":
                            continue
                        
                        if parent_span_id not in api_temp_bucket_dict[parent_span_api][parent_span][i]['span_index_dict'].keys():
                            continue

                        temp_index = api_temp_bucket_dict[parent_span_api][parent_span][i]['span_index_dict'][parent_span_id]
                        if api_pattern not in child_span_dict['upstream'].keys():
                            child_span_dict['upstream'][api_pattern] = {
                                'intensity': 0,
                                'duration': []
                            }
                        child_span_dict['upstream'][api_pattern]['intensity'] += 1
                        child_span_dict['upstream'][api_pattern]['duration'].append(api_temp_bucket_dict[parent_span_api][parent_span][i]['duration'][temp_index])
            
            # for api_pattern in api_pattern_list:                        
                for api_pod in self.config.data_dict['setting']['metric']['api_dict'][api_pattern]:
                    for feature_type in ['upstream', 'downstream']:
                        if api_pattern in child_span_dict[feature_type].keys():
                            value_dict = child_span_dict[feature_type][api_pattern]
                            api_feature_dict[f'<intensity>; {api_pattern}; {api_pod}; type: {feature_type}'].append(value_dict['intensity'])
                            if len(value_dict['duration']) > 0:
                                api_feature_dict[f'<duration>; {api_pattern}; {api_pod}; type: {feature_type}'].append(np.nan_to_num(np.nanmean(value_dict['duration'])))
                            else:
                                api_feature_dict[f'<duration>; {api_pattern}; {api_pod}; type: {feature_type}'].append(0.0)
                        else:
                            api_feature_dict[f'<intensity>; {api_pattern}; {api_pod}; type: {feature_type}'].append(0)
                            api_feature_dict[f'<duration>; {api_pattern}; {api_pod}; type: {feature_type}'].append(0.0)
                
        span_api_feature_df = pd.DataFrame(api_feature_dict)

        span_api_feature_df.to_csv(f'{result_base_path}/span_api_features.csv', index=False)
        
        # end api #######################
        
        
                
        span_feature_dict = {'timestamp': timestamp_list}

        for trace_pattern in trace_pattern_list:
            span_feature_dict[f'<intensity>; {trace_pattern}; type: upstream'] = []
            span_feature_dict[f'<duration>; {trace_pattern}; type: upstream'] = []
            span_feature_dict[f'<intensity>; {trace_pattern}; type: current'] = []
            span_feature_dict[f'<duration>; {trace_pattern}; type: current'] = []
            span_feature_dict[f'<intensity>; {trace_pattern}; type: downstream'] = []
            span_feature_dict[f'<duration>; {trace_pattern}; type: downstream'] = []

            for i in range(len(timestamp_list)):
                span_feature_dict[f'<intensity>; {trace_pattern}; type: current'].append(len(temp_bucket_dict[trace_pattern][i]['duration']))
                if len(temp_bucket_dict[trace_pattern][i]['duration']) > 0:
                    span_feature_dict[f'<duration>; {trace_pattern}; type: current'].append(np.nan_to_num(np.nanmean(temp_bucket_dict[trace_pattern][i]['duration'])))
                else:
                    span_feature_dict[f'<duration>; {trace_pattern}; type: current'].append(0.0)

        for i in range(len(timestamp_list)):
            child_span_dict = {
                'upstream': dict(),
                'downstream': dict()
            }
          
            for trace_pattern in trace_pattern_list:
                for index in range(len(temp_bucket_dict[trace_pattern][i]['parent_span'])):
                    parent_span_id = temp_bucket_dict[trace_pattern][i]['parent_span'][index]
                    if parent_span_id not in id_to_trace_pattern[i].keys():
                        continue
                    parent_span_pattern = id_to_trace_pattern[i][parent_span_id]
                    if parent_span_pattern not in child_span_dict['downstream'].keys():
                        child_span_dict['downstream'][parent_span_pattern] = {
                            'intensity': 0,
                            'duration': []
                        }
                    child_span_dict['downstream'][parent_span_pattern]['intensity'] += 1
                    child_span_dict['downstream'][parent_span_pattern]['duration'].append(
                        temp_bucket_dict[trace_pattern][i]['duration'][index])

                    if parent_span_id not in temp_bucket_dict[parent_span_pattern][i]['span_index_dict'].keys():
                        continue

                    temp_index = temp_bucket_dict[parent_span_pattern][i]['span_index_dict'][parent_span_id]
                    if trace_pattern not in child_span_dict['upstream'].keys():
                        child_span_dict['upstream'][trace_pattern] = {
                            'intensity': 0,
                            'duration': []
                        }
                    child_span_dict['upstream'][trace_pattern]['intensity'] += 1
                    child_span_dict['upstream'][trace_pattern]['duration'].append(temp_bucket_dict[parent_span_pattern][i]['duration'][temp_index])
                    
            for trace_pattern in trace_pattern_list:
                for feature_type in ['upstream', 'downstream']:
                    if trace_pattern in child_span_dict[feature_type].keys():
                        value_dict = child_span_dict[feature_type][trace_pattern]
                        span_feature_dict[f'<intensity>; {trace_pattern}; type: {feature_type}'].append(value_dict['intensity'])
                        if len(value_dict['duration']) > 0:
                            span_feature_dict[f'<duration>; {trace_pattern}; type: {feature_type}'].append(np.nan_to_num(np.nanmean(value_dict['duration'])))
                        else:
                            span_feature_dict[f'<duration>; {trace_pattern}; type: {feature_type}'].append(0.0)
                    else:
                        span_feature_dict[f'<intensity>; {trace_pattern}; type: {feature_type}'].append(0)
                        span_feature_dict[f'<duration>; {trace_pattern}; type: {feature_type}'].append(0.0)

        span_feature_df = pd.DataFrame(span_feature_dict)

        span_feature_df.to_csv(f'{result_base_path}/span_features.csv', index=False)

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
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    result_dict[dataset_type][date][cloud_bed] = dict()
                    trace_cloud_bed_path = f'{trace_date_path}/{cloud_bed}/raw_trace'

                    result_dict[dataset_type][date][cloud_bed]['span_features'] = pd.read_csv(f'{trace_cloud_bed_path}/span_features.csv')

        return result_dict
    
    def load_trace_api_csv(self):
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
    raw_trace_dao = RawTraceDao()
    # raw_trace_dao.extract_trace_patterns()
    raw_trace_dao.extract_trace_features()
    #feature_dict = raw_trace_dao.load_trace_csv()
    #api_feature_dict = raw_trace_dao.load_trace_api_csv()


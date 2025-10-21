import os
import sys
import pandas as pd

sys.path.append('/root/shared-nvme/work/code/Repdf/code')

import numpy as np
import json
import pickle
from tqdm import tqdm

from shared_util.common import *
from data_filter.CCF_AIOps_challenge_2022.base.base_generator import BaseGenerator
from data_filter.CCF_AIOps_challenge_2022.service.time_interval_label_generator import TimeIntervalLabelGenerator
from shared_util.file_handler import FileHandler

import random
prob = 0.9  # 70% be added in, 30% be skipped

# cd /root/shared-nvme/work/code/RCA/LasRCA/code
# nohup python -u ./data_filter/CCF_AIOps_challenge_2022/service/trace_generator.py > ./data_filter/CCF_AIOps_challenge_2022/service/trace_generator.log 2>&1 &


class ApiGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()
   
    def load_raw_trace_api(self):
        return self.raw_api_dao.load_api_csv()

    def calculate_trace_statistic(self):
        statistic_dict = dict()
        data_dict = dict()

        raw_data = self.load_raw_trace_api()
        file_dict = self.config.data_dict['file']
        for dataset_type, dataset_detail_dict in file_dict.items():
            if dataset_type == 'test':
                continue
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    feature_df = raw_data[dataset_type][date][cloud_bed]['span_api_features']
                    for feature_name in feature_df.keys():
                        if feature_name == 'timestamp':
                            continue
                        exact_feature_name = ApiGenerator.extract_entity_feature_name(feature_name)
                        if exact_feature_name not in data_dict.keys():
                            statistic_dict[exact_feature_name] = 0
                            data_dict[exact_feature_name] = []
                        data_dict[exact_feature_name].extend(raw_data[dataset_type][date][cloud_bed]['span_api_features'][feature_name].tolist())

        for feature_name in statistic_dict.keys():
            trace_data = data_dict[feature_name]
            median = np.nanmedian(trace_data)
            percentile_1 = np.nanpercentile(trace_data, 1)
            percentile_99 = np.nanpercentile(trace_data, 99)
            q1 = np.nanpercentile(trace_data, 25)
            q3 = np.nanpercentile(trace_data, 75)
            mean = np.nanmean(trace_data)
            std = np.nanstd(trace_data)
            valid_ratio = (np.count_nonzero(~np.isnan(trace_data))) / len(list(trace_data))

            statistic_dict[feature_name] = {
                'mean': mean,
                'std': std,
                'percentile_1': percentile_1,
                'q1': q1,
                'median': median,
                'q3': q3,
                'percentile_99': percentile_99,
                'valid_ratio': valid_ratio
            }

        folder = f'{self.config.param_dict["temp_data_storage"]}/analysis/trace'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/api_statistic.json', 'w') as f:
            json.dump(statistic_dict, f, indent=2)

    def z_score_trace_data(self):
        raw_data = self.load_raw_trace_api()

        file_dict = self.config.data_dict['file']
        with open(f'{self.config.param_dict["temp_data_storage"]}/analysis/trace/api_statistic.json', 'r') as f:
            statistic_dict = json.load(f)

        for dataset_type, dataset_detail_dict in file_dict.items():
            print(dataset_type)
            for date in dataset_detail_dict['date']:
                print(date)
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    print(cloud_bed)
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    feature_df = raw_data[dataset_type][date][cloud_bed]['span_api_features']
                    for feature_name in feature_df.keys():
                        if feature_name == 'timestamp':
                            continue
                        exact_feature_name = ApiGenerator.extract_entity_feature_name(feature_name)
                        print(exact_feature_name)
                        raw_trace_feature_data = raw_data[dataset_type][date][cloud_bed]['span_api_features'][feature_name]

                        iqr = statistic_dict[exact_feature_name]['q3'] - statistic_dict[exact_feature_name]['q1']
                        median = statistic_dict[exact_feature_name]['median']

                        # print(raw_data[dataset_type][date][cloud_bed]['span_api_features'][feature_name].dtype)
                        raw_data[dataset_type][date][cloud_bed]['span_api_features'][feature_name] = raw_data[dataset_type][date][cloud_bed]['span_api_features'][feature_name].astype(float)
                        
                        if iqr != 0:
                            update_trace_feature_data = (raw_trace_feature_data - median) / iqr
                            for i in range(len(update_trace_feature_data)):
                                
                                raw_data[dataset_type][date][cloud_bed]['span_api_features'].loc[i, feature_name] = update_trace_feature_data[i]

        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/api'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/all_api_features.pkl', 'wb') as f:
            pickle.dump(raw_data, f)

    def slice_api_features(self):
        result_dict = dict()

        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/trace'
        os.makedirs(folder, exist_ok=True)
        raw_trace_features = self.load_raw_trace_api()
        with open(f'{folder}/all_api_features.pkl', 'rb') as f:
            trace_features = pickle.load(f)
        container_list = self.config.data_dict['setting']['metric']['pod_order']
        file_dict = self.config.data_dict['file']
        for dataset_type, dataset_detail_dict in file_dict.items():
            result_dict[dataset_type] = dict()
            for date in dataset_detail_dict['date']:
                result_dict[dataset_type][date] = dict()
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    result_dict[dataset_type][date][cloud_bed] = {
                        'span_api_features': dict(),
                        'raw_span_api_features': dict()
                    }
                    feature_df = trace_features[dataset_type][date][cloud_bed]['span_api_features']
                    raw_feature_df = raw_trace_features[dataset_type][date][cloud_bed]['span_api_features']
                    for i in range(len(container_list)):
                        temp_dict, raw_temp_dict = {
                            'timestamp': feature_df['timestamp']
                        }, {
                            'timestamp': raw_feature_df['timestamp']
                        }
                        for feature_type in ['<intensity>', '<duration>']:
                            for feature_direction in ['upstream', 'current', 'downstream']:
                                feature_name = f'{feature_type}; cmdb_id: {container_list[i]}; type: {feature_direction}'
                                temp_dict[feature_name] = feature_df[feature_name]
                                raw_temp_dict[feature_name] = raw_feature_df[feature_name]
                        result_dict[dataset_type][date][cloud_bed]['span_api_features'][container_list[i]] = pd.DataFrame(temp_dict)
                        result_dict[dataset_type][date][cloud_bed]['raw_span_api_features'][container_list[i]] = pd.DataFrame(raw_temp_dict)

        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/trace'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/all_api.pkl', 'wb') as f:
            pickle.dump(result_dict, f)

    @staticmethod
    def extract_entity_feature_name(feature_name):
        cmdb_id = feature_name.split(';')[1].replace('2-0', '').replace('-0', '').replace('-1', '').replace('-2', '')
        return f'{feature_name.split(";")[0]};{cmdb_id};{feature_name.split(";")[2]}'

    def get_all_api(self) -> dict:
        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/api'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/all_api.pkl', 'rb') as f:
            api = pickle.load(f)
            return api
        
    def generate_api_data(self):
        all_api_dict = dict()
        with open(f'{self.config.param_dict["temp_data_storage"]}/dataset/api/all_api_features.pkl', 'rb') as f:
            temp_dict = pickle.load(f)
            for date_cloud_bed_data in temp_dict.values():
                for date, cloud_bed_data in date_cloud_bed_data.items():
                    all_api_dict[date] = cloud_bed_data

        def get_time_interval_trace_data(st, et, data_frame):
            return np.array(data_frame.query(f'{st} <= timestamp < {et}').iloc[:, data_frame.columns != "timestamp"].values)

        window_size_bar = tqdm(self.window_size_list)
        for window_size in window_size_bar:
            print()
            print("window_size: ", window_size)
            api_dict = dict()

            entity_features = []
            api_name_list = []
            record_features = True

            for node in self.config.data_dict['setting']['metric']['node_order']:
                entity_features.append((node, (0, 0)))

            # for service in self.config.data_dict['setting']['metric']['service_order']:
            #     entity_features.append((service, (0, 0)))
            # for data_type in ['train_valid', 'test']:
            #     print(data_type)
            #     time_interval_label_list = TimeIntervalLabelGenerator().get_time_interval_label(window_size)['time_interval'][data_type]
            #     api_dict[data_type] = []
            #     for time_interval in time_interval_label_list:
            #         print(time_interval)
            #         feature_index = 0
            #         api_data = all_api_dict[time_interval[0]][time_interval[1]]['span_api_features']

            #         temp = get_time_interval_trace_data(time_interval[2], time_interval[3], api_data)
            #         temp_name_list = list(api_data.columns[api_data.columns != "timestamp"])

            #         data = []
            #         for service in self.config.data_dict['setting']['metric']['service_order']:
            #             pod_list = ApiGenerator.rename_service2pod(service)
            #             for pod in pod_list:
            #                 # 此处需仔细思考该如何处理，需不需要针对api_pattern进行拆解获得内部的pod信息？
            #                 pod_related_api_name_list = []
            #                 for api_pattern in api_pattern_list:
            #                     for position_type in ['upstream', 'current', 'downstream']:
            #                         for feature_type in ['<intensity>', '<duration>']:
            #                             api_feature_name = f'{feature_type}; {api_pattern}; {pod}; type: {position_type}'
            #                             if api_feature_name not in temp_name_list:
            #                                 continue
            #                             data.append(temp[:, temp_name_list.index(api_feature_name)])
            #                             pod_related_api_name_list.append(api_feature_name)
            #                 if record_features:
            #                     entity_features.append((pod, (feature_index, feature_index + len(pod_related_api_name_list))))
            #                     feature_index += len(pod_related_api_name_list)
            #                     api_name_list.extend(pod_related_api_name_list)

            #         api_dict[data_type].append(np.array(data).transpose())
            #         record_features = False
                
            for data_type in ['train_valid', 'test']:
                print(data_type)
                time_interval_label_list = TimeIntervalLabelGenerator().get_time_interval_label(window_size)['time_interval'][data_type]
                api_dict[data_type] = []
                for time_interval in time_interval_label_list:
                    print(time_interval)
                    feature_index = 0
                    api_data = all_api_dict[time_interval[0]][time_interval[1]]['span_api_features']

                    temp = get_time_interval_trace_data(time_interval[2], time_interval[3], api_data)
                    temp_name_list = list(api_data.columns[api_data.columns != "timestamp"])

                    data = []
                    counter = 0
                    for service in self.config.data_dict['setting']['metric']['service_order']:
                        api_pattern_list = ApiGenerator.rename_service2api(service)
                        service_related_api_name_list = []
                        for api_pattern in api_pattern_list:
                            for position_type in ['upstream', 'current', 'downstream']:
                                for feature_type in ['<intensity>', '<duration>']:
                                    api_feature_name = f'{feature_type}; {api_pattern}; type: {position_type}'
                                    if api_feature_name not in temp_name_list:
                                        continue
                                    # if random.random() < prob and counter <= 115:
                                    counter += 1
                                    data.append(temp[:, temp_name_list.index(api_feature_name)])
                                    service_related_api_name_list.append(api_feature_name)
                        if record_features:
                            entity_features.append((service, (feature_index, feature_index + len(service_related_api_name_list))))
                            feature_index += len(service_related_api_name_list)
                            api_name_list.extend(service_related_api_name_list)

                    api_dict[data_type].append(np.array(data).transpose())
                    record_features = False
                    
            for pod in self.config.data_dict['setting']['metric']['pod_order']:
                entity_features.append((pod, (0, 0)))

            folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/api'
            with open(f'{folder}/api_window_size_{window_size}.pkl', 'wb') as f:
                pickle.dump({
                    'api_data': api_dict,
                    'entity_features': entity_features,
                    'api_names': api_name_list
                }, f)
            window_size_bar.set_description("Api dataset generating".format(window_size))
        
    def get_api(self, window_size) -> dict:
        folder = f'{self.config.param_dict["temp_data_storage"]}/dataset/api'
        with open(f'{folder}/api_window_size_{window_size}.pkl', 'rb') as f:
            api = pickle.load(f)
            return api


if __name__ == '__main__':
    api_generator = ApiGenerator()
    # api_generator.calculate_trace_statistic()
    # api_generator.z_score_trace_data()
    api_generator.generate_api_data()
    # test = api_generator.get_all_api()

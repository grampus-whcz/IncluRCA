import numpy as np
import json
import pickle
from tqdm import tqdm

from shared_util.file_handler import FileHandler
from shared_util.time_handler import TimeHandler
from data_filter.CCF_AIOps_challenge_2022.base.base_generator import BaseGenerator


class TimeIntervalLabelGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()

    def get_ground_truth(self, ground_truth_dict, index):
        return {
            'timestamp': ground_truth_dict['timestamp'][index],
            'level': ground_truth_dict['level'][index],
            'cmdb_id': ground_truth_dict['cmdb_id'][index],
            'fault_type': self.fault_type_list.index(ground_truth_dict['failure_type'][index]) + 1
        }

    @staticmethod
    def get_date_timestamp_list(date_str: str) -> list:
        date_start_timestamp = TimeHandler.datetime_to_timestamp(date_str + ' 00:00:00')
        return list(range(date_start_timestamp, date_start_timestamp + 24 * 60 * 60, 60))

    def get_ground_truth_label(self, ground_truth):
        label = np.zeros(len(self.all_entity_list))
        label[self.all_entity_list.index(ground_truth['cmdb_id'])] = ground_truth['fault_type']
        if ground_truth['level'] == 'service':
            label[self.all_entity_list.index(f'{ground_truth["cmdb_id"]}-0')] = ground_truth['fault_type']
            label[self.all_entity_list.index(f'{ground_truth["cmdb_id"]}-1')] = ground_truth['fault_type']
            label[self.all_entity_list.index(f'{ground_truth["cmdb_id"]}-2')] = ground_truth['fault_type']
            label[self.all_entity_list.index(f'{ground_truth["cmdb_id"]}2-0')] = ground_truth['fault_type']
        return label
    
    def get_normal_ground_truth_label(self):
        label = np.zeros(len(self.all_entity_list))
        return label

    def slice_ground_truth_timestamp(self, date, cloud_bed, ground_truth_timestamp, window_size, sliding_ratio):
        interval_list = []
        start_timestamp = TimeHandler.datetime_to_timestamp(date + ' 00:00:00')
        c_ts = ground_truth_timestamp - (ground_truth_timestamp - start_timestamp) % 60
        s_ts = c_ts - int(window_size * sliding_ratio) * 60
        e_ts = s_ts + window_size * 60
        interval_list.append((date, cloud_bed, s_ts, e_ts))
        return interval_list
    
    def slice_normal_ground_truth_timestamp(self, dataset_type, date, cloud_bed, ground_truth_timestamp, window_size, sliding_ratio):
        interval_list = []
        # 1. 先基于原始date计算出「目标时间的时分秒对应的偏移量」（核心：提取时间部分，剥离日期影响）
        date_start_timestamp = TimeHandler.datetime_to_timestamp(date + ' 00:00:00')
        # 计算ground_truth_timestamp在date这一天内的偏移秒数（即当天00:00:00到该时间的秒数，对应时分秒信息）
        day_offset_seconds = (ground_truth_timestamp - date_start_timestamp) % (24 * 60 * 60)
        # 2. 计算基于原始date的c_ts、s_ts、e_ts（仅用于获取时间窗口的偏移关系，不保留其日期）
        c_ts = ground_truth_timestamp - (ground_truth_timestamp - date_start_timestamp) % 60
        s_ts = c_ts - int(window_size * sliding_ratio) * 60
        e_ts = s_ts + window_size * 60
        # 计算s_ts和e_ts相对于c_ts的时间偏移（或直接基于day_offset_seconds重新计算目标日期的时间戳）
        # 更简洁的方式：直接基于2022-03-19的0点时间戳 + 对应偏移秒数，得到目标日期的时间戳
        target_date = '2022-03-19'
        target_date_start_timestamp = TimeHandler.datetime_to_timestamp(target_date + ' 00:00:00')
        
        # 计算目标日期（2022-03-19）对应的c_ts、s_ts、e_ts
        # 先得到目标日期的c_ts（对齐分钟，与原始时间的时分秒一致）
        target_c_ts = target_date_start_timestamp + (day_offset_seconds - day_offset_seconds % 60)
        # 基于目标c_ts计算目标s_ts和e_ts（保持窗口大小和滑动比例不变）
        target_s_ts = target_c_ts - int(window_size * sliding_ratio) * 60
        target_e_ts = target_s_ts + window_size * 60
        
        # 3. 添加目标日期（2022-03-19）的时间间隔，而非原始date
        if dataset_type == 'train_valid':
            interval_list.append((target_date, cloud_bed, target_s_ts, target_e_ts))
        else:
            interval_list.append((target_date, 'cloudbed-1', target_s_ts, target_e_ts))
        return interval_list

    def generate_time_interval_label(self):
        window_size_bar = tqdm(self.window_size_list)
        for window_size in window_size_bar:
            faulty_time_interval, faulty_y = {'train_valid': [], 'test': []}, {'train_valid': [], 'test': []}
            normal_time_interval, normal_y = {'train_valid': [], 'test': []}, {'train_valid': [], 'test': []}
            faulty_entity_type, faulty_template, faulty_cmdb_id, faulty_root_cause_type = {'train_valid': [], 'test': []}, {'train_valid': [], 'test': []}, {'train_valid': [], 'test': []}, {'train_valid': [], 'test': []}

            for dataset_type in ['train_valid', 'test']:
                print("dataset_type: ", dataset_type)
                train_ground_truth_timestamp_dict = dict()
                for date, cloud_dict in self.ground_truth_dao.get_ground_truth(dataset_type).items():
                    print("date: ", date)
                    for cloud_bed in cloud_dict.keys():
                        train_ground_truth_timestamp_dict[f'{date}/{cloud_bed}'] = []

                        for i in range(len(cloud_dict[cloud_bed]['timestamp'])):
                            ground_truth = self.get_ground_truth(cloud_dict[cloud_bed], i)
                            train_ground_truth_timestamp_dict[f'{date}/{cloud_bed}'].append(ground_truth['timestamp'])
                            temp_time_interval_list = self.slice_ground_truth_timestamp(date, cloud_bed, ground_truth['timestamp'], window_size, 0.5)
                            faulty_time_interval[dataset_type].extend(temp_time_interval_list)
                            faulty_y[dataset_type].extend([self.get_ground_truth_label(ground_truth) for i in range(len(temp_time_interval_list))])
                            normal_temp_time_interval_list = self.slice_normal_ground_truth_timestamp(dataset_type, date, cloud_bed, ground_truth['timestamp'], window_size, 0.5)
                            normal_time_interval[dataset_type].extend(normal_temp_time_interval_list)
                            normal_y[dataset_type].extend([self.get_normal_ground_truth_label() for i in range(len(normal_temp_time_interval_list))])
                            faulty_entity_type[dataset_type].append(ground_truth['level'])
                            faulty_template[dataset_type].append(ground_truth['cmdb_id'].replace('2-0', '').replace('-0', '').replace('-1', '').replace('-2', '').replace('-3', '').replace('-4', '').replace('-5', '').replace('-6', ''))
                            faulty_cmdb_id[dataset_type].append(ground_truth['cmdb_id'])
                            faulty_root_cause_type[dataset_type].append(self.fault_type_list[ground_truth['fault_type'] - 1])

            folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/time_interval_and_label')
            with open(f'{folder}/time_interval_window_size_{window_size}.pkl', 'wb') as f:
                pickle.dump({
                    'time_interval': {
                        'train_valid': faulty_time_interval['train_valid'],
                        'normal_for_train_valid': normal_time_interval['train_valid'],
                        'test': faulty_time_interval['test'],
                        'normal_for_test': normal_time_interval['test'],
                    },
                    'y': {
                        'train_valid': faulty_y['train_valid'],
                        'normal_for_train_valid': normal_y['train_valid'],
                        'test': faulty_y['test'],
                        'normal_for_test': normal_y['test']
                    },
                    'entity_type': {
                        'train_valid': faulty_entity_type['train_valid'],
                        'test': faulty_entity_type['test']
                    },
                    'template': {
                        'train_valid': faulty_template['train_valid'],
                        'test': faulty_template['test']
                    },
                    'cmdb_id': {
                        'train_valid': faulty_cmdb_id['train_valid'],
                        'test': faulty_cmdb_id['test']
                    },
                    'root_cause_type': {
                        'train_valid': faulty_root_cause_type['train_valid'],
                        'test': faulty_root_cause_type['test']
                    }
                }, f)
            window_size_bar.set_description("Time interval and label generating".format(window_size))

    def get_time_interval_label(self, window_size) -> dict:
        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/time_interval_and_label')
        with open(f'{folder}/time_interval_window_size_{window_size}.pkl', 'rb') as f:
            time_interval_label = pickle.load(f)
            return time_interval_label


if __name__ == '__main__':
    time_interval_label_generator = TimeIntervalLabelGenerator()
    time_interval_label_generator.generate_time_interval_label()
    # time_interval_label_generator.get_time_interval_label(9)

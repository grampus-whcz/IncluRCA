import sys

sys.path.append('/root/shared-nvme/work/code/Repdf/code')

import os
import pandas as pd
import numpy as np
from data_filter.CCF_AIOps_challenge_2022.base.base_class import BaseClass
from shared_util.file_handler import FileHandler
from shared_util.time_handler import TimeHandler
import json
import pickle
import re
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


class RawLogDao(BaseClass):
    def __init__(self):
        super().__init__()
        self.log_pattern_dict = dict()
        self.log_template_miner_dict = dict()
        self.istio_words = []

    def init_template_miner(self):
        drain_config = TemplateMinerConfig()
        drain_config.load(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'drain3.ini'))
        drain_config.profiling_enabled = True
        for language in set(self.config.data_dict['setting']['service_knowledge']['service_language'].values()):
            self.log_template_miner_dict[language] = TemplateMiner(config=drain_config)
            self.log_pattern_dict[language] = []

    def extract_container_log_patterns(self):
        self.init_template_miner()
        file_dict = self.config.data_dict['file']
        result_base_path = f'{self.config.param_dict["temp_data_storage"]}/analysis'

        for dataset_type, dataset_detail_dict in file_dict.items():
            if dataset_type == 'test':
                continue
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    data_base_path = f'{dataset_detail_dict["base_folder"]}/{date}/{cloud_bed}/log/all/log_filebeat-testbed-log-service.csv'
                    reader = pd.read_csv(data_base_path, chunksize=100000)
                    for chunk in reader:
                        log_df = chunk
                        for _, row in log_df.iterrows():
                            is_valid, log_pattern = self.parse_template(row)
                            if not is_valid:
                                continue
                            service_language = self.config.data_dict['setting']['service_knowledge']['service_language'][RawLogDao.rename_pod2service(row['cmdb_id'])]
                            if log_pattern not in self.log_pattern_dict[service_language]:
                                self.log_pattern_dict[service_language].append(log_pattern)
                    
                    # 处理istio日志。istio日志的处理与容器日志类似，但它将被处理两次：一次用于提取日志模式，另一次用于提取词频。      
                    data_base_path = f'{dataset_detail_dict["base_folder"]}/{date}/{cloud_bed}/log/all/log_filebeat-testbed-log-envoy.csv'
                    reader = pd.read_csv(data_base_path, chunksize=100000)
                    for chunk in reader:
                        log_df = chunk
                        for _, row in log_df.iterrows():
                            is_valid, log_pattern = self.parse_template(row)
                            if not is_valid:
                                continue
                            service_language = self.config.data_dict['setting']['service_knowledge']['service_language'][RawLogDao.rename_pod2service(row['cmdb_id'])]
                            if log_pattern not in self.log_pattern_dict[service_language]:
                                self.log_pattern_dict[service_language].append(log_pattern)

        with open(FileHandler.set_folder(f'{result_base_path}/log') + '/log_istio_patterns.json', 'w') as f:
            json.dump(self.log_pattern_dict, f, indent=2)

        with open(FileHandler.set_folder(f'{result_base_path}/log') + '/log_istio_template_miner.pkl', 'wb') as f:
            pickle.dump(self.log_template_miner_dict, f)

        self.logger.debug('Already extract log patterns.')

    def parse_template(self, row):
        is_valid, log_pattern = True, ''
        if row['cmdb_id'] not in self.all_entity_list:
            is_valid = False
        elif not isinstance(row['value'], str):
            is_valid = False
        else:
            service_language = self.config.data_dict['setting']['service_knowledge']['service_language'][RawLogDao.rename_pod2service(row['cmdb_id'])]
            log_pattern = self.log_template_miner_dict[service_language].add_log_message(row['value'])['template_mined']
        return is_valid, log_pattern

    def calculate_word_count(self, row):
        stopwords_regex_list = [
            " \"*([0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3})\\:?([0-9]{1,5})?\"*",  # 匹配IPv4地址（如172.20.8.79），并可选地匹配其后的端口号（如:50051）
            "\"(\\w|\\d)*-(\\w|\\d)*-(\\w|\\d)*-(\\w|\\d)*-(\\w|\\d)*\"",                   # 匹配类似于UUID的字符串，例如c165805c-950d-98c2-990a-325a3bcd1f7c
            "( |^)\"*(GET|POST|PUT|DELETE)\"*",                                             # 匹配常见的HTTP请求方法：GET、POST、PUT、DELETE， ( |^)确保该方法出现在行首或者前面有一个空格。
            " \"*[0-9]+\"*",                                                                # 匹配由一个或多个数字组成的字符串，通常用于匹配状态码、时间等数值
            " (\"*grpc-(\\w)*/)[0-9]+.[0-9]+.[0-9]+\"*",                                    # 匹配gRPC客户端的用户代理字符串，比如grpc-go/1.31.0
            " (\"*Go(-(\\w)*)*/)[0-9]+(.[0-9]+)*\"*",                                       # 类似于gRPC用户代理的匹配规则，但专用于Go语言的HTTP库，Go(-(\\w)*)*/ 可能匹配像Go-http-client/这样的字符串，[0-9]+(.[0-9]+)* 匹配版本号格式
            "( |^)\"*-\"*",                                                                 # 简单匹配单独的破折号-，可能用于日志中某些字段为空时的占位符
            " (default|(\\(manylinux; chttp2\\))|(HTTP/\\d(.\\d)*\"*))\"*"                  # 匹配“default”、“(manylinux; chttp2)”或者以“HTTP/”开头的HTTP版本号（如HTTP/2）
        ]
        is_valid, word_dict = True, dict()
        service_cmdb_id = RawLogDao.rename_pod2service(row['cmdb_id'])
        if service_cmdb_id not in self.all_entity_list:
            is_valid = False
        elif not isinstance(row['value'], str):
            is_valid = False
        else:
            new_str = row['value'].strip()
            for stop_word in stopwords_regex_list:
                new_str = re.sub(stop_word, '', new_str)
            for word in new_str.split(' '):
                if word != '':
                    if word not in word_dict.keys():
                        word_dict[word] = 0
                    word_dict[word] += 1
        return is_valid, word_dict

    def extract_istio_words(self):
        file_dict = self.config.data_dict['file']
        result_base_path = f'{self.config.param_dict["temp_data_storage"]}/analysis'

        result_dict = {
            'word_set': []
        }
        for dataset_type, dataset_detail_dict in file_dict.items():
            if dataset_type == 'test':
                continue
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    data_base_path = f'{dataset_detail_dict["base_folder"]}/{date}/{cloud_bed}/log/all/log_filebeat-testbed-log-envoy.csv'
                    reader = pd.read_csv(data_base_path, chunksize=100000)
                    for chunk in reader:
                        log_df = chunk
                        for _, row in log_df.iterrows():
                            is_valid, word_dict = self.calculate_word_count(row)
                            if not is_valid:
                                continue
                            for word in word_dict.keys():
                                if word not in result_dict['word_set']:
                                    result_dict['word_set'].append(word)

        with open(FileHandler.set_folder(f'{result_base_path}/log') + '/istio_words.json', 'w') as f:
            json.dump(result_dict, f, indent=2)
        self.logger.debug('Already extract istio_words.')

    def extract_log_features(self):
        self.init_template_miner()
        analysis_base_path = f'{self.config.param_dict["temp_data_storage"]}/analysis'
        with open(FileHandler.set_folder(f'{analysis_base_path}/log') + '/log_patterns.json') as f:
            self.log_pattern_dict = json.load(f)

        with open(FileHandler.set_folder(f'{analysis_base_path}/log') + '/istio_words.json') as f:
            self.istio_words = json.load(f)['word_set']

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
                    result_cloud_bed_path = FileHandler.set_folder(f'{result_date_path}/{cloud_bed}/raw_log')
                    data_base_path = f'{dataset_detail_dict["base_folder"]}/{date}/{cloud_bed}/log/all/log_filebeat-testbed-log-service.csv'
                    self.logger.debug(f'Preprocessing container logs: dataset_type: {dataset_type}, date: {date}, cloudbed: {cloud_bed}.')
                    self.process_container_logs(timestamp_list, data_base_path, result_cloud_bed_path, 'service_log_patterns_count')
                    
                    # # 处理istio日志。istio日志的处理与容器日志类似，但它将被处理两次：一次用于提取日志模式，另一次用于提取词频。
                    data_base_path = f'{dataset_detail_dict["base_folder"]}/{date}/{cloud_bed}/log/all/log_filebeat-testbed-log-envoy.csv'
                    self.logger.debug(f'Preprocessing container logs: dataset_type: {dataset_type}, date: {date}, cloudbed: {cloud_bed}.')
                    self.process_container_logs(timestamp_list, data_base_path, result_cloud_bed_path, 'istio_log_patterns_count')

                    data_base_path = f'{dataset_detail_dict["base_folder"]}/{date}/{cloud_bed}/log/all/log_filebeat-testbed-log-envoy.csv'
                    self.logger.debug(f'Preprocessing istio logs: dataset_type: {dataset_type}, date: {date}, cloudbed: {cloud_bed}.')
                    self.process_istio_logs(timestamp_list, data_base_path, result_cloud_bed_path)

    def process_container_logs(self, timestamp_list: list, data_base_path: str, result_base_path: str, file_name: str):
        result_dict = {'timestamp': timestamp_list}
        for service in self.config.data_dict['setting']['metric']['service_order']:
            service_language = self.config.data_dict['setting']['service_knowledge']['service_language'][service]
            pod_list = RawLogDao.rename_service2pod(service)
            for pod in pod_list:
                for i in range(len(self.log_pattern_dict[service_language])):
                    result_dict[f'{pod}; <log pattern {i}>'] = [0 for _ in timestamp_list]

        reader = pd.read_csv(data_base_path, chunksize=100000)
        chunk_num = 0
        for chunk in reader:
            
            chunk_num += 1
            self.logger.debug(f"Processing container chunk {chunk_num}")
            
            log_df = chunk
            for _, row in log_df.iterrows():
                index = int((row["timestamp"] - timestamp_list[0]) / 60)
                is_valid, log_pattern = self.parse_template(row)
                if not is_valid:
                    continue
                service_language = self.config.data_dict['setting']['service_knowledge']['service_language'][RawLogDao.rename_pod2service(row['cmdb_id'])]
                if log_pattern in self.log_pattern_dict[service_language]:
                    result_dict[f'{row["cmdb_id"]}; <log pattern {self.log_pattern_dict[service_language].index(log_pattern)}>'][index] += 1

        result_df = pd.DataFrame(result_dict)
        result_df.to_csv(f'{result_base_path}/{file_name}.csv', index=False) ##log_patterns_count

    def process_istio_logs(self, timestamp_list: list, data_base_path: str, result_base_path: str):
        result_dict = {
            'timestamp': timestamp_list,
            'total': [0 for _ in timestamp_list]
        }
        for word in self.istio_words:
            if 'latency=' in word or 'ttl=' in word:
                continue
            result_dict[f'{word}'] = [0 for _ in timestamp_list]
        for service in self.config.data_dict['setting']['metric']['service_order']:
            pod_list = RawLogDao.rename_service2pod(service)
            for pod in pod_list:
                for word in self.istio_words:
                    if 'latency=' in word or 'ttl=' in word:
                        continue
                    result_dict[f'{pod}; {word}'] = [0 for _ in timestamp_list]

        reader = pd.read_csv(data_base_path, chunksize=100000)
        chunk_num = 0
        for chunk in reader:
            
            chunk_num += 1
            self.logger.debug(f"Processing container chunk {chunk_num}")
            
            log_df = chunk
            for _, row in log_df.iterrows():
                index = int((row["timestamp"] - timestamp_list[0]) / 60)
                is_valid, word_dict = self.calculate_word_count(row)
                if not is_valid or row["cmdb_id"] not in self.config.data_dict['setting']['metric']['pod_order']:
                    continue
                for word, count in word_dict.items():
                    if word not in self.istio_words or 'latency=' in word or 'ttl=' in word:
                        continue
                    result_dict['total'][index] += count
                    result_dict[f'{word}'][index] += count
                    result_dict[f'{row["cmdb_id"]}; {word}'][index] += count

        result_df = pd.DataFrame(result_dict)
        result_df.to_csv(f'{result_base_path}/istio_word_count.csv', index=False)

    def load_log_csv(self, log_type):
        result_dict = dict()

        file_dict = self.config.data_dict['file']
        log_base_path = f'{self.config.param_dict["temp_data_storage"]}/raw_data'

        for dataset_type, dataset_detail_dict in file_dict.items():
            result_dict[dataset_type] = dict()
            log_dataset_type_path = f'{log_base_path}/{dataset_type}'
            for date in dataset_detail_dict['date']:
                result_dict[dataset_type][date] = dict()
                log_date_path = f'{log_dataset_type_path}/{date}'
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    result_dict[dataset_type][date][cloud_bed] = dict()
                    log_cloud_bed_path = f'{log_date_path}/{cloud_bed}/raw_log'

                    if log_type == 'container':
                        result_dict[dataset_type][date][cloud_bed]['container_log_features'] = pd.read_csv(f'{log_cloud_bed_path}/log_patterns_count.csv')
                    elif log_type == 'istio':
                        result_dict[dataset_type][date][cloud_bed]['istio_log_features'] = pd.read_csv(f'{log_cloud_bed_path}/istio_word_count.csv')

        return result_dict
    
    # istio's and service's log patterns hardly have two respective patterns that are exactly the same. so just merge them like filling in the blanks which is 0 in the two tables.
    # for service_log_patterns_count.csv and istio_log_patterns_count.csv, merge them and produce final log_patterns_count.csv
    # important
    # line 270 log_patterns_count.csv can be replaced by service_log_patterns_count.csv, so service part of log only contains service,
    # istio information only in istio words pattern.
    def merged_log_patterns_count(self):
        file_dict = self.config.data_dict['file']
        log_base_path = f'{self.config.param_dict["temp_data_storage"]}/raw_data'

        for dataset_type, dataset_detail_dict in file_dict.items():
            log_dataset_type_path = f'{log_base_path}/{dataset_type}'
            for date in dataset_detail_dict['date']: # ['2022-03-19']
                log_date_path = f'{log_dataset_type_path}/{date}'
                for cloud_bed in dataset_detail_dict['cloud_bed']: # ['cloudbed-1']
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    log_cloud_bed_path = f'{log_date_path}/{cloud_bed}/raw_log'
                    # 读取两个CSV文件
                    df_istio = pd.read_csv(f'{log_cloud_bed_path}/istio_log_patterns_count.csv')
                    df_service = pd.read_csv(f'{log_cloud_bed_path}/service_log_patterns_count.csv')

                    # 确保结构一致（列名、行数）
                    assert df_istio.shape == df_service.shape, "两个DataFrame的形状不一致"
                    assert all(df_istio.columns == df_service.columns), "两个DataFrame的列名不一致"

                    # 将数据转换为numpy数组进行高效操作
                    arr_istio = df_istio.to_numpy()
                    arr_service = df_service.to_numpy()

                    # 创建一个结果数组
                    result = np.where(
                        (arr_istio == 0) & (arr_service == 0),
                        0,
                        np.maximum(arr_istio, arr_service)
                    )

                    # 将结果转为DataFrame并保存
                    result_df = pd.DataFrame(result, columns=df_istio.columns, index=df_istio.index)

                    # 可选：保存到新的CSV文件
                    result_df.to_csv(f'{log_cloud_bed_path}/log_patterns_count.csv', index=False)

                    print("合并完成，结果已保存为 log_patterns_count.csv")
        


if __name__ == '__main__':
    raw_log_dao = RawLogDao()
    # raw_log_dao.extract_container_log_patterns()
    # raw_log_dao.extract_istio_words()
    # raw_log_dao.extract_log_features()
    # raw_log_dao.merged_log_patterns_count()

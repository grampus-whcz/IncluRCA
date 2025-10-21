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


class RawLogDaoTest(BaseClass):
    def __init__(self):
        super().__init__()
        self.log_pattern_dict = dict()
        self.log_template_miner_dict = dict()
        self.istio_words = []
    
    def calculate_word_count(self, row):
        stopwords_regex_list = [
            " \"*([0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3})\\:?([0-9]{1,5})?\"*",  # 匹配IPv4地址（如172.20.8.79），并可选地匹配其后的端口号（如:50051）
            "\"(\\w|\\d)*-(\\w|\\d)*-(\\w|\\d)*-(\\w|\\d)*-(\\w|\\d)*\"",                   # 匹配类似于UUID的字符串，例如c165805c-950d-98c2-990a-325a3bcd1f7c
            "( |^)\"*(GET|POST|PUT|DELETE)\"*",                                             # 匹配常见的HTTP请求方法：GET、POST、PUT、DELETE， ( |^)确保该方法出现在行首或者前面有一个空格。
            " \"*[0-9]+\"*",                                                                # 匹配由一个或多个数字组成的字符串，通常用于匹配状态码、时间等数值
            " (\"*grpc-(\\w)*/)[0-9]+.[0-9]+.[0-9]+\"*",                                    # 匹配gRPC客户端的用户代理字符串，比如grpc-go/1.31.0
            " (\"*Go(-(\\w)*)*/)[0-9]+(.[0-9]+)*\"*",                                       # 类似于gRPC用户代理的匹配规则，但专用于Go语言的HTTP库，Go(-(\\w)*)*/ 可能匹配像Go-http-client/这样的字符串，[0-9]+(.[0-9]+)* 匹配版本号格式
            "( |^)\"*-\"*",                                                                 # 简单匹配单独的破折号-，可能用于日志中某些字段为空时的占位符
            " (default|(\\(manylinux; chttp2\\))|(HTTP/\\d(.\\d)*\"*))\"*",                 # 匹配“default”、“(manylinux; chttp2)”或者以“HTTP/”开头的HTTP版本号（如HTTP/2）
            "ttl=[0-9]+h[0-9]+m[0-9]+\\.[0-9]+s",
            "latency=[0-9]+\\.[0-9]+ms"
        ]
        is_valid, word_dict = True, dict()
        service_cmdb_id = RawLogDaoTest.rename_pod2service(row['cmdb_id'])
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
            if dataset_type == 'test' or dataset_type == 'train_valid':
                continue
            for date in ['2022-03-19']:
                for cloud_bed in ['cloudbed-1']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    data_base_path = f'{dataset_detail_dict["base_folder"]}/{date}/{cloud_bed}/log/all/e.csv'
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

        with open(FileHandler.set_folder(f'{result_base_path}/log') + '/test_istio_words.json', 'w') as f:
            json.dump(result_dict, f, indent=2)
        self.logger.debug('Already extract istio_words.')
        
if __name__ == '__main__':
    raw_log_dao = RawLogDaoTest()
    raw_log_dao.extract_istio_words()
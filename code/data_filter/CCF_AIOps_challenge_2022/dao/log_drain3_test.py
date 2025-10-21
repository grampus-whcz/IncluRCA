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
            if dataset_type == 'test' or dataset_type == 'train_valid':
                continue
            for date in ['2022-03-19']:
                for cloud_bed in ['cloudbed-1']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    data_base_path = f'{dataset_detail_dict["base_folder"]}/{date}/{cloud_bed}/log/all/g.csv'
                    reader = pd.read_csv(data_base_path, chunksize=100000)
                    for chunk in reader:
                        log_df = chunk
                        for _, row in log_df.iterrows():
                            is_valid, log_pattern = self.parse_template(row)
                            if not is_valid:
                                continue
                            service_language = self.config.data_dict['setting']['service_knowledge']['service_language'][RawLogDaoTest.rename_pod2service(row['cmdb_id'])]
                            if log_pattern not in self.log_pattern_dict[service_language]:
                                self.log_pattern_dict[service_language].append(log_pattern)

        with open(FileHandler.set_folder(f'{result_base_path}/log') + '/test_log_patterns.json', 'w') as f:
            json.dump(self.log_pattern_dict, f, indent=2)

        with open(FileHandler.set_folder(f'{result_base_path}/log') + '/test_log_template_miner.pkl', 'wb') as f:
            pickle.dump(self.log_template_miner_dict, f)

        self.logger.debug('Already extract log patterns.')

    def parse_template(self, row):
        is_valid, log_pattern = True, ''
        if row['cmdb_id'] not in self.all_entity_list:
            is_valid = False
        elif not isinstance(row['value'], str):
            is_valid = False
        else:
            service_language = self.config.data_dict['setting']['service_knowledge']['service_language'][RawLogDaoTest.rename_pod2service(row['cmdb_id'])]
            log_pattern = self.log_template_miner_dict[service_language].add_log_message(row['value'])['template_mined']
        return is_valid, log_pattern

    


if __name__ == '__main__':
    raw_log_dao_test = RawLogDaoTest()
    raw_log_dao_test.extract_container_log_patterns()

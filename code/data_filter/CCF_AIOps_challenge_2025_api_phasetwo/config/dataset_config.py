import os
import json


class DataConfig:
    def __init__(self):
        self.data_dict = dict()
        self.param_dict = dict()

        self.set_data_dict()
        self.set_param_dict()

    def set_data_dict(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metric_setting.json')) as f:
            metric_setting_dict = json.load(f)

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'service_knowledge.json')) as f:
            service_knowledge_dict = json.load(f)

        data_base_path = '/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone'
        self.data_dict = {
            'file': {
                'normal': {
                    'base_folder': data_base_path,
                    'date': [],
                    'cloud_bed': ['cloudbed']
                },
                'train_valid': {
                    'base_folder': data_base_path,
                    'date': ['2025-06-06', '2025-06-07', '2025-06-08','2025-06-09', '2025-06-10', '2025-06-11'],
                    'cloud_bed': ['cloudbed']
                },
                # CST 2025-06-12 00:00:00--2025-06-12 23:59:59 
                # UTC 2025-06-11 16:00:00--2025-06-12 15:59:59
                # groud_truth 切分时间 UTC 2025-06-11 16:00:00
                'test': {
                    'base_folder': data_base_path,
                    'date': ['2025-06-12', '2025-06-13', '2025-06-14'],
                    'cloud_bed': ['cloudbed'],
                }
            },
            'ground_truth': {
                'train_valid': data_base_path + '/phase1.jsonl',
                'test': data_base_path + '/phase1.jsonl'
            },
            'setting': {
                'metric': metric_setting_dict,
                'service_knowledge': service_knowledge_dict
            }
        }

    def set_param_dict(self):
        data_base_path = '/root/shared-nvme/work/code/RCA/IncluRCA'
        self.param_dict = {
            'logging': {
                'level': 'DEBUG'
            },
            'temp_data_storage': f'{data_base_path}/temp_data/2025_CCF_AIOps_challenge'
        }

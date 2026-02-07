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

        data_base_path = '/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phasetwo'
        self.data_dict = {
            'file': {
                'normal': {
                    'base_folder': data_base_path,
                    'date': ['2025-06-29'],
                    'cloud_bed': ['cloudbed']
                },
                'train_valid': {
                    'base_folder': data_base_path,
                    'date': ['2025-06-17', '2025-06-18', '2025-06-19','2025-06-20', '2025-06-21'],
                    'cloud_bed': ['cloudbed']
                },
                # CST 2025-06-12 00:00:00--2025-06-12 23:59:59 
                # UTC 2025-06-11 16:00:00--2025-06-12 15:59:59
                # groud_truth 切分时间 UTC 2025-06-11 16:00:00
                'test': {
                    'base_folder': data_base_path,
                    'date': ['2025-06-24', '2025-06-27', '2025-06-28'],
                    'cloud_bed': ['cloudbed'],
                }
            },
            'ground_truth': {
                'train_valid': data_base_path + '/phase2.jsonl',
                'test': data_base_path + '/phase2.jsonl'
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
            'temp_data_storage': f'{data_base_path}/temp_data/2025_CCF_AIOps_challenge_phasetwo'
        }

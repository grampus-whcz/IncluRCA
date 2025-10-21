from abc import ABC
from data_filter.CCF_AIOps_challenge_2025.config.dataset_config import DataConfig
from shared_util.logger import Logger
from datetime import datetime, timedelta


class BaseClass(ABC):
    def __init__(self):
        self.config = DataConfig()
        self.logger = Logger(self.config.param_dict['logging']['level']).logger

        self.all_entity_list = []
        self.all_entity_list.extend(self.config.data_dict['setting']['metric']['node_order'])
        self.all_entity_list.extend(self.config.data_dict['setting']['metric']['service_order'])
        self.all_entity_list.extend(self.config.data_dict['setting']['metric']['tidb_order'])
        self.all_entity_list.extend(self.config.data_dict['setting']['metric']['pod_order'])
        

    @staticmethod
    def rename_pod2service(pod_name):
        return pod_name.replace('-0', '').replace('-1', '').replace('-2', '')

    @staticmethod
    def rename_service2pod(service_name):
        if service_name == "redis-cart":
            return [f'{service_name}-0']
        else:
            return [f'{service_name}-0', f'{service_name}-1', f'{service_name}-2']


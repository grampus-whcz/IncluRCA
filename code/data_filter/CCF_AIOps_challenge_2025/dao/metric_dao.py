import sys
import os

sys.path.append('/root/shared-nvme/work/code/RCA/IncluRCA/code')

import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

from data_filter.CCF_AIOps_challenge_2025.base.base_class import BaseClass
from shared_util.file_handler import FileHandler
from shared_util.time_handler import TimeHandler
from datetime import datetime, timedelta
import pyarrow.parquet as pq
import pyarrow as pa
from collections import defaultdict


class RawMetricDao(BaseClass):
    def __init__(self):
        super().__init__()

    def process_node_metric(self, timestamp_list: list, data_base_path: str, result_base_path: str, date: str):
        node_list = self.config.data_dict['setting']['metric']['node_order']
        metric_info_dict = self.config.data_dict['setting']['metric']['related_metrics']['node']
        
        metric_info_dir = f'{data_base_path}/infra/infra_node/'
        result_base_path = FileHandler.set_folder(f'{result_base_path}/node')

        metric_df = {}
        for metric_type, metric_name_list in metric_info_dict.items():
            for raw_metric_name in metric_name_list:
                                
                data_base_path = metric_info_dir + f"infra_node_{raw_metric_name}_{date}.parquet"
                table = pq.read_table(data_base_path)
                batches = table.to_batches(max_chunksize=65536)               
                for batch in batches:
                    for record in batch.to_pylist():
                        timestamp_tmp = TimeHandler.datetime_to_timestamp_new1(record.get("time"))
                        node_tmp = record.get("kubernetes_node")                            
                        value = record.get(raw_metric_name)
                        key = (timestamp_tmp, node_tmp, raw_metric_name)
                        metric_df[key] = value 
                        
        node_list_bar = tqdm(node_list)
        for node in node_list_bar:
            node_metric_dict = {
                'timestamp': timestamp_list
            } 
            for raw_metric_name in metric_name_list:
                metric_name = raw_metric_name
                node_metric_dict[metric_name] = []                            
                
                for timestamp in timestamp_list:
                    if (timestamp, node, raw_metric_name) in metric_df:
                        node_metric_dict[metric_name].append(metric_df.get((timestamp, node, raw_metric_name)))
                    else:
                        node_metric_dict[metric_name].append(np.nan)
            node_df = pd.DataFrame(node_metric_dict)
            node_df.to_csv(f'{result_base_path}/{node}.csv', index=False)
            node_list_bar.set_description("Node metric csv generating".format(node))

    def process_container_metric(self, timestamp_list: list, data_base_path: str, result_base_path: str, date: str):
        pod_list = self.config.data_dict['setting']['metric']['pod_order']
        metric_info_dict = self.config.data_dict['setting']['metric']['related_metrics']['container']

        metric_info_dir1 = f'{data_base_path}/infra/infra_pod/'
        metric_info_dir2 = f'{data_base_path}/apm/pod/'
        result_base_path = FileHandler.set_folder(f'{result_base_path}/container')
        
        metric_df = {}
        # infra/infra_pod
        for metric_type, metric_name_list in metric_info_dict.items():
            if metric_type == "Infra":
                for raw_metric_name in metric_name_list:
                                    
                    data_base_path = metric_info_dir1 + f"infra_pod_{raw_metric_name}_{date}.parquet"
                    table = pq.read_table(data_base_path)
                    batches = table.to_batches(max_chunksize=65536)               
                    for batch in batches:
                        for record in batch.to_pylist():
                            timestamp_tmp = TimeHandler.datetime_to_timestamp_new1(record.get("time"))
                            pod_tmp = record.get("pod")                            
                            value = record.get(raw_metric_name)
                            key = (timestamp_tmp, pod_tmp, raw_metric_name)
                            metric_df[key] = value
        
        # apm/pod
        pod_list_bar = tqdm(pod_list)
        # 初始化为嵌套 defaultdict
        metric_apm_df = defaultdict(dict)
        for pod in pod_list_bar:
            data_base_path = metric_info_dir2 + f"pod_{pod}_{date}.parquet"
            if not os.path.isdir(metric_info_dir2):
                print(f"目录不存在，跳过: {metric_info_dir2}")
                pass
            elif not os.path.exists(data_base_path):
                print(f"文件不存在，跳过: {data_base_path}")
                pass
            else:
                table = pq.read_table(data_base_path)
                batches = table.to_batches(max_chunksize=65536)               
                for batch in batches:
                    for record in batch.to_pylist():
                        timestamp_tmp = TimeHandler.datetime_to_timestamp_new1(record.get("time"))
                        pod_tmp = record.get("object_id")
                        request = record.get("request")
                        response = record.get("response")
                        rrt = record.get("rrt")
                        rrt_max = record.get("rrt_max")
                        error = record.get("error")
                        client_error = record.get("client_error")
                        server_error = record.get("server_error")
                        timeout = record.get("timeout")
                        error_ratio = record.get("error_ratio")
                        client_error_ratio = record.get("client_error_ratio")
                        server_error_ratio = record.get("server_error_ratio")
                        key = (timestamp_tmp, pod_tmp)
                        metric_apm_df[key]["request"] = request
                        metric_apm_df[key]["response"] = response
                        metric_apm_df[key]["rrt"] = rrt
                        metric_apm_df[key]["rrt_max"] = rrt_max
                        metric_apm_df[key]["error"] = error
                        metric_apm_df[key]["client_error"] = client_error
                        metric_apm_df[key]["server_error"] = server_error
                        metric_apm_df[key]["timeout"] = timeout
                        metric_apm_df[key]["error_ratio"] = error_ratio
                        metric_apm_df[key]["client_error_ratio"] = client_error_ratio
                        metric_apm_df[key]["server_error_ratio"] = server_error_ratio
                        
        pod_list_bar = tqdm(pod_list)
        for pod in pod_list_bar:
            pod_metric_dict = {
                'timestamp': timestamp_list
            }
            for metric_type, metric_name_list in metric_info_dict.items():
                if metric_type == "Infra":
                    for raw_metric_name in metric_name_list:
                        metric_name = raw_metric_name
                        pod_metric_dict[metric_name] = []                            
                        
                        for timestamp in timestamp_list:
                            if (timestamp, pod, raw_metric_name) in metric_df:
                                pod_metric_dict[metric_name].append(metric_df.get((timestamp, pod, raw_metric_name)))
                            else:
                                pod_metric_dict[metric_name].append(np.nan)
                else:
                    for raw_metric_name in metric_name_list:
                        metric_name = raw_metric_name
                        pod_metric_dict[metric_name] = []                            
                        
                        for timestamp in timestamp_list:
                            if (timestamp, pod) in metric_apm_df:
                                pod_metric_dict[metric_name].append(metric_apm_df.get((timestamp, pod)).get(metric_name))
                            else:
                                pod_metric_dict[metric_name].append(np.nan)
            pod_df = pd.DataFrame(pod_metric_dict)
            pod_df.to_csv(f'{result_base_path}/{pod}.csv', index=False)
            pod_list_bar.set_description("Pod metric csv generating".format(pod))

    def process_tidb_metric(self, timestamp_list: list, data_base_path: str, result_base_path: str, date: str):
        tidb_list = self.config.data_dict['setting']['metric']['tidb_order']
        metric_info_dict = self.config.data_dict['setting']['metric']['related_metrics']['tidb']

        metric_info_dir1 = f'{data_base_path}/infra/infra_tidb/'
        metric_info_dir2 = f'{data_base_path}/other/'
        result_base_path = FileHandler.set_folder(f'{result_base_path}/tidb')

        metric_df = {}
        for metric_type, metric_name_list in metric_info_dict.items():
            if metric_type == "tidb":
                for raw_metric_name in metric_name_list:
                                    
                    data_base_path = metric_info_dir1 + f"infra_tidb_{raw_metric_name}_{date}.parquet"
                    if not os.path.isdir(metric_info_dir1):
                        print(f"目录不存在，跳过: {metric_info_dir1}")
                        pass
                    elif not os.path.exists(data_base_path):
                        print(f"文件不存在，跳过: {data_base_path}")
                        pass
                    else:
                        table = pq.read_table(data_base_path)
                        batches = table.to_batches(max_chunksize=65536)               
                        for batch in batches:
                            for record in batch.to_pylist():
                                timestamp_tmp = TimeHandler.datetime_to_timestamp_new1(record.get("time"))
                                value = record.get(raw_metric_name)
                                key = (timestamp_tmp, "tidb", raw_metric_name)
                                metric_df[key] = value
            elif metric_type == "tikv":
                for raw_metric_name in metric_name_list:
                                    
                    data_base_path = metric_info_dir2 + f"infra_tikv_{raw_metric_name}_{date}.parquet"
                    if not os.path.isdir(metric_info_dir2):
                        print(f"目录不存在，跳过: {metric_info_dir2}")
                        pass
                    elif not os.path.exists(data_base_path):
                        print(f"文件不存在，跳过: {data_base_path}")
                        pass
                    else:
                        table = pq.read_table(data_base_path)
                        batches = table.to_batches(max_chunksize=65536)               
                        for batch in batches:
                            for record in batch.to_pylist():
                                timestamp_tmp = TimeHandler.datetime_to_timestamp_new1(record.get("time"))
                                value = record.get(raw_metric_name)
                                key = (timestamp_tmp, "tikv", raw_metric_name)
                                metric_df[key] = value
            elif metric_type == "pd":
                for raw_metric_name in metric_name_list:
                                    
                    data_base_path = metric_info_dir2 + f"infra_pd_{raw_metric_name}_{date}.parquet"
                    if not os.path.isdir(metric_info_dir2):
                        print(f"目录不存在，跳过: {metric_info_dir2}")
                        pass
                    elif not os.path.exists(data_base_path):
                        print(f"文件不存在，跳过: {data_base_path}")
                        pass
                    else:
                        table = pq.read_table(data_base_path)
                        batches = table.to_batches(max_chunksize=65536)               
                        for batch in batches:
                            for record in batch.to_pylist():
                                timestamp_tmp = TimeHandler.datetime_to_timestamp_new1(record.get("time"))
                                value = record.get(raw_metric_name)
                                key = (timestamp_tmp, "pd", raw_metric_name)
                                metric_df[key] = value
        
        for metric_type, metric_name_list in metric_info_dict.items():
            tidb_metric_dict = {
                'timestamp': timestamp_list
            }
            if metric_type == "tidb":
                for raw_metric_name in metric_name_list:
                    metric_name = raw_metric_name
                    tidb_metric_dict[metric_name] = []                            
                    
                    for timestamp in timestamp_list:
                        if (timestamp, metric_type, raw_metric_name) in metric_df:
                            tidb_metric_dict[metric_name].append(metric_df.get((timestamp, metric_type, raw_metric_name)))
                        else:
                            tidb_metric_dict[metric_name].append(np.nan)
                tidb_df = pd.DataFrame(tidb_metric_dict)
                tidb_df.to_csv(f'{result_base_path}/tidb-{metric_type}.csv', index=False)
            if metric_type == "tikv":
                for raw_metric_name in metric_name_list:
                    metric_name = raw_metric_name
                    tidb_metric_dict[metric_name] = []                            
                    
                    for timestamp in timestamp_list:
                        if (timestamp, metric_type, raw_metric_name) in metric_df:
                            tidb_metric_dict[metric_name].append(metric_df.get((timestamp, metric_type, raw_metric_name)))
                        else:
                            tidb_metric_dict[metric_name].append(np.nan)
                tidb_df = pd.DataFrame(tidb_metric_dict)
                tidb_df.to_csv(f'{result_base_path}/tidb-{metric_type}.csv', index=False)
            if metric_type == "pd":
                for raw_metric_name in metric_name_list:
                    metric_name = raw_metric_name
                    tidb_metric_dict[metric_name] = []                            
                    
                    for timestamp in timestamp_list:
                        if (timestamp, metric_type, raw_metric_name) in metric_df:
                            tidb_metric_dict[metric_name].append(metric_df.get((timestamp, metric_type, raw_metric_name)))
                        else:
                            tidb_metric_dict[metric_name].append(np.nan)
                tidb_df = pd.DataFrame(tidb_metric_dict)
                tidb_df.to_csv(f'{result_base_path}/tidb-{metric_type}.csv', index=False)

    def process_service_metric(self, timestamp_list: list, data_base_path: str, result_base_path: str, date: str):
        service_list = self.config.data_dict['setting']['metric']['service_order']
        metric_info_dict = self.config.data_dict['setting']['metric']['related_metrics']['service']

        metric_info_dir = f'{data_base_path}/apm/service/'
        result_base_path = FileHandler.set_folder(f'{result_base_path}/service')
        
        
        # apm/service
        service_list_bar = tqdm(service_list)
        # 初始化为嵌套 defaultdict
        metric_apm_df = defaultdict(dict)
        for pod in service_list_bar:
            data_base_path = metric_info_dir + f"service_{pod}_{date}.parquet"
            if not os.path.isdir(metric_info_dir):
                print(f"目录不存在，跳过: {metric_info_dir}")
                pass
            elif not os.path.exists(data_base_path):
                print(f"文件不存在，跳过: {data_base_path}")
                pass
            else:
                table = pq.read_table(data_base_path)
                batches = table.to_batches(max_chunksize=65536)               
                for batch in batches:
                    for record in batch.to_pylist():
                        timestamp_tmp = TimeHandler.datetime_to_timestamp_new1(record.get("time"))
                        pod_tmp = record.get("object_id")
                        request = record.get("request")
                        response = record.get("response")
                        rrt = record.get("rrt")
                        rrt_max = record.get("rrt_max")
                        error = record.get("error")
                        client_error = record.get("client_error")
                        server_error = record.get("server_error")
                        timeout = record.get("timeout")
                        error_ratio = record.get("error_ratio")
                        client_error_ratio = record.get("client_error_ratio")
                        server_error_ratio = record.get("server_error_ratio")
                        key = (timestamp_tmp, pod_tmp)
                        metric_apm_df[key]["request"] = request
                        metric_apm_df[key]["response"] = response
                        metric_apm_df[key]["rrt"] = rrt
                        metric_apm_df[key]["rrt_max"] = rrt_max
                        metric_apm_df[key]["error"] = error
                        metric_apm_df[key]["client_error"] = client_error
                        metric_apm_df[key]["server_error"] = server_error
                        metric_apm_df[key]["timeout"] = timeout
                        metric_apm_df[key]["error_ratio"] = error_ratio
                        metric_apm_df[key]["client_error_ratio"] = client_error_ratio
                        metric_apm_df[key]["server_error_ratio"] = server_error_ratio
                        
        service_list_bar = tqdm(service_list)
        for service in service_list_bar:
            service_metric_dict = {
                'timestamp': timestamp_list
            }
            for metric_type, metric_name_list in metric_info_dict.items():
                if metric_type == "APM":
                    for raw_metric_name in metric_name_list:
                        metric_name = raw_metric_name
                        service_metric_dict[metric_name] = []                            
                        
                        for timestamp in timestamp_list:
                            if (timestamp, service) in metric_apm_df:
                                service_metric_dict[metric_name].append(metric_apm_df.get((timestamp, service)).get(metric_name))
                            else:
                                service_metric_dict[metric_name].append(np.nan)
            service_df = pd.DataFrame(service_metric_dict)
            service_df.to_csv(f'{result_base_path}/{service}.csv', index=False)
            service_list_bar.set_description("Service metric csv generating".format(service))
        
    def generate_metric_csv(self):
        file_dict = self.config.data_dict['file']
        result_base_path = f'{self.config.param_dict["temp_data_storage"]}/raw_data'

        for dataset_type, dataset_detail_dict in file_dict.items():
            result_dataset_type_path = FileHandler.set_folder(f'{result_base_path}/{dataset_type}')
            for date in dataset_detail_dict['date']:
                result_date_path = FileHandler.set_folder(f'{result_dataset_type_path}/{date}')
                timestamp_list = [TimeHandler.datetime_to_timestamp(f'{date} 00:00:00') + i * 60 for i in range(0, 24 * 60)]
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    
                    result_cloud_bed_path = FileHandler.set_folder(f'{result_date_path}/{cloud_bed}/raw_metric')
                    data_base_path = f'{dataset_detail_dict["base_folder"]}/{date}/{cloud_bed}/metric-parquet'

                    for resource_type in ['node', 'container', 'service', 'tidb']:
                        self.logger.debug(f'Preprocessing metrics: dataset_type: {dataset_type}, date: {date}, '
                                          f'cloudbed: {cloud_bed}, resource_type: {resource_type}.')
                        eval(f'self.process_{resource_type}_metric(timestamp_list, data_base_path, result_cloud_bed_path, date)')

    def load_metric_csv(self):
        result_dict = dict()

        file_dict = self.config.data_dict['file']
        metric_base_path = f'{self.config.param_dict["temp_data_storage"]}/raw_data'

        for dataset_type, dataset_detail_dict in file_dict.items():
            result_dict[dataset_type] = dict()
            metric_dataset_type_path = f'{metric_base_path}/{dataset_type}'
            for date in dataset_detail_dict['date']:
                result_dict[dataset_type][date] = dict()
                metric_date_path = f'{metric_dataset_type_path}/{date}'
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    
                    result_dict[dataset_type][date][cloud_bed] = dict()
                    metric_cloud_bed_path = f'{metric_date_path}/{cloud_bed}/raw_metric'
                    for resource_type in ['node', 'container', 'service', 'tidb']:
                        result_dict[dataset_type][date][cloud_bed][resource_type] = dict()
                        entity_list = self.get_entity_list(resource_type)
                        for entity in entity_list:
                            result_dict[dataset_type][date][cloud_bed][resource_type][entity] = pd.read_csv(f'{metric_cloud_bed_path}/{resource_type}/{entity}.csv').interpolate()
        return result_dict

    def get_entity_list(self, resource_type: str):
        entity_list = []
        if resource_type == 'node':
            entity_list = self.config.data_dict['setting']['metric']['node_order']
        elif resource_type == 'container':
            entity_list = self.config.data_dict['setting']['metric']['pod_order']
        elif resource_type == 'service':
            entity_list = self.config.data_dict['setting']['metric']['service_order']
        elif resource_type == 'tidb':
            entity_list = self.config.data_dict['setting']['metric']['tidb_order']
        return entity_list
    
    def test(self):
        timestamp_list = [TimeHandler.datetime_to_timestamp((datetime.strptime("2025-06-06", "%Y-%m-%d") - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")) + i * 60 for i in range(0, 24 * 60)]
        # self.process_node_metric(timestamp_list, "/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/2025-06-06/cloudbed/metric-parquet", 
        #                          "/root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2025_CCF_AIOps_challenge/raw_data/train_valid/2025-06-06/cloudbed/raw_metric", 
        #                          "2025-06-06")
        
        # self.process_container_metric(timestamp_list, "/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/2025-06-06/cloudbed/metric-parquet", 
        #                          "/root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2025_CCF_AIOps_challenge/raw_data/train_valid/2025-06-06/cloudbed/raw_metric", 
        #                          "2025-06-06")
        
        # self.process_service_metric(timestamp_list, "/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/2025-06-06/cloudbed/metric-parquet", 
        #                          "/root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2025_CCF_AIOps_challenge/raw_data/train_valid/2025-06-06/cloudbed/raw_metric", 
        #                          "2025-06-06")
        
        self.process_tidb_metric(timestamp_list, "/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/2025-06-06/cloudbed/metric-parquet", 
                                 "/root/shared-nvme/work/code/RCA/IncluRCA/temp_data/2025_CCF_AIOps_challenge/raw_data/train_valid/2025-06-06/cloudbed/raw_metric", 
                                 "2025-06-06")
        
        

if __name__ == '__main__':
    raw_metric_dao = RawMetricDao()
    raw_metric_dao.generate_metric_csv()
    # raw_metric_dao.test()

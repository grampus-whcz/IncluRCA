from data_filter.CCF_AIOps_challenge_2025.base.base_class import BaseClass
from data_filter.CCF_AIOps_challenge_2025.dao.metric_dao import RawMetricDao
from data_filter.CCF_AIOps_challenge_2025.dao.trace_dao import RawTraceDao
from data_filter.CCF_AIOps_challenge_2025.dao.log_dao import RawLogDao
from data_filter.CCF_AIOps_challenge_2025.dao.topology_dao import TopologyDao
from data_filter.CCF_AIOps_challenge_2025.dao.ground_truth_dao import GroundTruthDao


class BaseGenerator(BaseClass):
    def __init__(self):
        super().__init__()
        self.window_size_list = [7, 9, 11, 13, 15, 17]
        self.ground_truth_dao = GroundTruthDao()
        self.raw_metric_dao = RawMetricDao()
        self.raw_trace_dao = RawTraceDao()
        self.raw_log_dao = RawLogDao()
        self.topology_dao = TopologyDao()
        self.fault_type_list = [
            'jvm cpu',        # 0         # jvm fault
            'jvm exception',  # 1
            'jvm gc',         # 2
            'jvm latency',    # 3
            'network corrupt',# 4          # network attack
            'network delay',  # 5
            'network loss',   # 6
            'node cpu',       # 7           # node fault
            'node disk fill', # 8
            'node memory',    # 9
            'pod failure',    #10           # pod fault
            'pod kill',       #11
            'cpu stress',     #12           # stress test
            'memory stress'   #13
        ]
        self.fault_type_related_o11y_names = {
            
            0:{
            "exact": ["client_error+", "max_rrt+", "pod_cpu_usage", "pod_network_transmit_packets", "rrt", "rrt+", "latency_anomalies", "error", "exception", "failed"],
            "fuzzy": []
            },
            1:{
            "exact": ["error", "exception", "failed", "stall"],
            "fuzzy": []
            },
            2:{
            "exact": ["error", "exception", "failed", "stall"],
            "fuzzy": []
            },
            3:{
            "exact": ["client_error_ratio", "error_ratio", "rrt", "latency_anomalies", "error", "exception", "failed"],
            "fuzzy": []
            },
            4:{
            "exact": ["client_error_ratio", "error_ratio", "max_rrt+", "request-", "response-", "rrt", "rrt+", "timeout", "timeout+", "latency_anomalies", "request_proportion_anomalies", "error", "exception", "failed", "timeout"],
            "fuzzy": []
            },
            5:{
            "exact": ["client_error+", "max_rrt+", "request-", "response-", "rrt", "rrt+", "latency_anomalies", "error", "exception", "failed"],
            "fuzzy": []
            },
            6:{
            "exact": ["max_rrt+", "request-", "response-", "rrt", "rrt+", "latency_anomalies", "request_proportion_anomalies", "abort", "disconnect", "error", "exception", "failed", "retry", "timeout"],
            "fuzzy": []
            },
            7:{
            "exact": ["node_cpu_usage_rate"],
            "fuzzy": []
            },
            8:{
            "exact": ["node_cpu_usage_rate", "node_filesystem_usage_rate"],
            "fuzzy": []
            },
            9:{
            "exact": ["node_memory_usage_rate", "latency_anomalies", "error", "failed"],
            "fuzzy": []
            },
            10:{
            "exact": ["error+", "max_rrt+", "request-", "response-", "rrt", "rrt+", "server_error+", "timeout", "timeout+", "latency_anomalies", "request_proportion_anomalies", "error", "exception", "failed", "refused", "unavailable"],
            "fuzzy": []
            },
            11:{
            "exact": ["client_error+", "request-", "response-", "timeout+", "latency_anomalies", "request_proportion_anomalies"],
            "fuzzy": []
            },
            12:{
            "exact": ["max_rrt+", "pod_cpu_usage", "pod_processes", "request-", "response-", "rrt", "rrt+", "timeout+", "latency_anomalies"],
            "fuzzy": []
            },
            13:{
            "exact": ["client_error+", "pod_network_receive_bytes", "pod_network_receive_packets", "pod_network_transmit_bytes", "pod_network_transmit_packets", "pod_processes", "request", "response", "latency_anomalies", "request_proportion_anomalies", "error"],
            "fuzzy": []
            }
        }
        self.offset = 20

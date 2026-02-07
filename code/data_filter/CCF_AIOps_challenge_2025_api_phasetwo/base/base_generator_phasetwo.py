from data_filter.CCF_AIOps_challenge_2025_api.base.base_class import BaseClass
from data_filter.CCF_AIOps_challenge_2025_api.dao.metric_dao import RawMetricDao
from data_filter.CCF_AIOps_challenge_2025_api.dao.trace_dao import RawTraceDao
from data_filter.CCF_AIOps_challenge_2025_api.dao.log_dao import RawLogDao
from data_filter.CCF_AIOps_challenge_2025_api.dao.topology_dao import TopologyDao
from data_filter.CCF_AIOps_challenge_2025_api.dao.ground_truth_dao import GroundTruthDao
from data_filter.CCF_AIOps_challenge_2025_api.dao.api_dao import RawApiDao

# Note: for phase two, the groudtruth file "phase2.jsonl" should be modified to a new form that tidb should be fit for our data preprocessing progress. 
# Through carefully reviewed, we found that tidb-tikv-0, tidb-tidb-0, tidb-pd-0 do not appear in the log, trace or metric. So these so called pod name just appears in the "phase2.jsonl".
# These label names do not actually serve any purpose.
# 
# To maintain consistency with the business logic in Phase One, we treat TiDB as a new entity type named "tidb," which comprises three sub-entities: 
# "tidb-tikv," "tidb-tidb," and "tidb-pd." Accordingly, in "phase2.jsonl," we rename "tidb-tikv-0," "tidb-tidb-0," and "tidb-pd-0" to "tidb-tikv," "tidb-tidb," and "tidb-pd," respectively, 
# and change their "instance_type" from "pod" to "tidb." This approach ensures that our "metric_setting.json" requires no modifications and remains compatible with Phase Two.

class BaseGenerator(BaseClass):
    def __init__(self):
        super().__init__()
        self.window_size_list = [7, 9, 11, 13, 15, 17]
        self.ground_truth_dao = GroundTruthDao()
        self.raw_metric_dao = RawMetricDao()
        self.raw_trace_dao = RawTraceDao()
        self.raw_log_dao = RawLogDao()
        self.topology_dao = TopologyDao()
        self.raw_api_dao = RawApiDao()
        self.fault_type_list = [
            'dns error',                # 0          # dns fault
            'code error',               # 1          # erroneous change
            'io fault',                 # 2          # io fault
            'target port misconfig',    # 3          # misconfiguration
            'network corrupt',          # 4          # network attack
            'network delay',            # 5
            'network loss',             # 6
            'node cpu',                 # 7           # node fault
            'node disk fill',           # 8
            'node memory',              # 9
            'pod failure',              #10           # pod fault
            
            'cpu stress',               #11           # stress test
            'memory stress'             #12
        ]
        self.fault_type_related_o11y_names = {            
            0:{
            "exact": ["error_ratio", "rrt", "server_error_ratio", "latency_anomalies", "error", "exception", "failed", "misbehaving", "timeout", "unavailable"],
            "fuzzy": []
            },
            1:{
            "exact": ["client_error_ratio", "error_ratio", "rrt", "timeout", "latency_anomalies", "request_proportion_anomalies", "error", "exception", "failed"],
            "fuzzy": []
            },
            2:{
            "exact": ["tikv_cpu_usage", "tikv_read_mbps", "tikv_snapshot_apply_count", "tikv_write_wal_mbps", "latency_anomalies"],
            "fuzzy": []
            },
            3:{
            "exact": ["client_error_ratio", "error_ratio", "latency_anomalies", "request_proportion_anomalies", "error", "exception", "failed", "refused", "unavailable"],
            "fuzzy": []
            },
            4:{
            "exact": ["rrt", "latency_anomalies", "request_proportion_anomalies", "error", "exception", "failed", "timeout", "unavailable"],
            "fuzzy": []
            },
            5:{
            "exact": ["client_error_ratio", "error_ratio", "rrt", "timeout", "latency_anomalies", "error", "exception"],
            "fuzzy": []
            },
            6:{
            "exact": ["client_error_ratio", "error_ratio", "rrt", "timeout", "latency_anomalies", "request_proportion_anomalies", "abort", "disconnect", "error", "exception", "failed", "retry", "timeout"],
            "fuzzy": []
            },
            7:{
            "exact": ["node_cpu_usage_rate", "request_proportion_anomalies"],
            "fuzzy": []
            },
            8:{
            "exact": ["latency_anomalies", "node_filesystem_usage_rate"],
            "fuzzy": []
            },
            9:{
            "exact": ["node_memory_usage_rate", "latency_anomalies", "request_proportion_anomalies", "error", "exception", "failed", "refused", "timeout"],
            "fuzzy": []
            },
            10:{
            "exact": ["latency_anomalies", "request_proportion_anomalies", "error", "failed", "refused", "timeout", "unavailable"],
            "fuzzy": []
            },
            11:{
            "exact": ["pod_cpu_usage", "pod_processes", "rrt", "timeout", "latency_anomalies", "error", "exception"],
            "fuzzy": []
            },
            12:{
            "exact": ["client_error_ratio", "error_ratio", "error", "exception", "failed"],
            "fuzzy": []
            }
        }
        self.offset = 20

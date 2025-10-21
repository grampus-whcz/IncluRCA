from data_filter.CCF_AIOps_challenge_2022.base.base_class import BaseClass
from data_filter.CCF_AIOps_challenge_2022.dao.metric_dao import RawMetricDao
from data_filter.CCF_AIOps_challenge_2022.dao.trace_dao import RawTraceDao
from data_filter.CCF_AIOps_challenge_2022.dao.log_dao import RawLogDao
from data_filter.CCF_AIOps_challenge_2022.dao.topology_dao import TopologyDao
from data_filter.CCF_AIOps_challenge_2022.dao.ground_truth_dao import GroundTruthDao


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
            'node节点CPU故障',              # 0
            'node节点CPU爬升',              # 1
            'node 内存消耗',                # 2
            'node 磁盘读IO消耗',            # 3
            'node 磁盘写IO消耗',            # 4
            'node 磁盘空间消耗',            # 5
            'k8s容器cpu负载',               # 6 service/pod
            'k8s容器内存负载',              # 7 service/pod
            'k8s容器网络延迟',              # 8 service/pod
            'k8s容器网络丢包',              # 9 service/pod
            'k8s容器网络资源包损坏',        # 10 service/pod
            'k8s容器网络资源包重复发送',    # 11 service/pod
            'k8s容器读io负载',             # 12 service/pod
            'k8s容器写io负载',             # 13 service/pod
            'k8s容器进程中止'              # 14 service/pod
        ]
        self.en_fault_type_list = [
            'Node CPU Failure',
            'Node CPU Climb',
            'Node Memory Consumption',
            'Node Disk Read IO Consumption',
            'Node Disk Write IO Consumption',
            'Node Disk Space Consumption',
            'K8s CPU Load',
            'K8s Memory Load',
            'K8s Network Delay',
            'K8s Network Packet Loss',
            'K8s Network Packet Corruption',
            'K8s Network Packet Duplication',
            'K8s Disk Read IO Load',
            'K8s Disk Write IO Load',
            'K8s Process Terminated'
        ]
        self.fault_type_related_o11y_names = {
            0: {
                "exact": ["system.load.1", "system.load.15", "system.load.5", "system.cpu.iowait", "system.cpu.pct_usage", "system.cpu.system", "system.cpu.user", "system.os.nofile.current", "system.os.nofile.used_pct"],
                "fuzzy": []
            },
            1: {
                "exact": ["system.load.1", "system.load.15", "system.load.5", "system.cpu.iowait", "system.cpu.pct_usage", "system.cpu.system", "system.cpu.user", "system.os.nofile.current", "system.os.nofile.used_pct"],
                "fuzzy": []
            },
            2: {
                "exact": ["system.mem.pct_usage", "system.mem.real.pct_useage", "system.mem.real.used", "system.mem.usable", "system.mem.used", "system.os.nofile.current", "system.os.nofile.used_pct"],
                "fuzzy": []
            },
            3: {
                "exact": ["system.io.avg_q_sz", "system.io.await", "system.io.r_await", "system.io.r_s", "system.io.rkb_s", "system.io.svctm", "system.io.util", "system.disk.free", "system.disk.pct_usage", "system.disk.used", "system.os.nofile.current", "system.os.nofile.used_pct"],
                "fuzzy": []
            },
            4: {
                "exact": ["system.io.avg_q_sz", "system.io.await", "system.io.w_await", "system.io.w_s", "system.io.svctm", "system.io.util", "system.disk.free", "system.disk.pct_usage", "system.disk.used", "system.os.nofile.current", "system.os.nofile.used_pct"],
                "fuzzy": []
            },
            5: {
                "exact": ["system.disk.free", "system.disk.pct_usage", "system.disk.used", "system.disk.free", "system.disk.pct_usage", "system.disk.used", "system.os.nofile.current", "system.os.nofile.used_pct"],
                "fuzzy": []
            },
            6: {
                "exact": ["kpi_container_cpu_usage_seconds", "kpi_container_cpu_user_seconds", "kpi_container_cpu_system_seconds", "kpi_container_cpu_cfs_throttled_seconds", "kpi_container_cpu_cfs_throttled_periods", "kpi_container_cpu_cfs_periods"],
                "fuzzy": []
            },
            7: {
                "exact": ["kpi_container_memory_cache", "kpi_container_memory_mapped_file", "kpi_container_memory_usage_MB", "kpi_container_memory_working_set_MB", "kpi_container_memory_rss", "kpi_container_memory_failures", "kpi_container_memory_failcnt", "kpi_container_memory_max_usage_MB"],
                "fuzzy": []
            },
            8: {
                "exact": ["kpi_istio_tcp_received_bytes", "kpi_istio_tcp_sent_bytes", "kpi_container_sockets"],
                "fuzzy": ["<intensity>", "<duration>"]
            },
            9: {
                "exact": ["kpi_container_network_receive_packets", "kpi_container_network_receive_MB", "kpi_container_network_transmit_packets", "kpi_container_network_transmit_MB", "kpi_istio_tcp_received_bytes", "kpi_istio_tcp_sent_bytes", "kpi_container_sockets"],
                "fuzzy": []
            },
            10: {
                "exact": ["kpi_istio_tcp_received_bytes", "kpi_istio_tcp_sent_bytes", "kpi_container_sockets"],
                "fuzzy": ["<intensity>", "<duration>", "http2.remote_reset"]
            },
            11: {
                "exact": ["kpi_container_network_receive_packets", "kpi_container_network_receive_MB", "kpi_container_network_transmit_packets", "kpi_container_network_transmit_MB", "kpi_istio_tcp_received_bytes", "kpi_istio_tcp_sent_bytes", "kpi_container_sockets"],
                "fuzzy": []
            },
            12: {
                "exact": ["kpi_container_fs_inodes", "kpi_container_fs_reads_MB", "kpi_container_fs_usage_MB", "kpi_container_fs_reads"],
                "fuzzy": []
            },
            13: {
                "exact": ["kpi_container_fs_inodes", "kpi_container_fs_writes_MB", "kpi_container_fs_usage_MB", "kpi_container_fs_writes"],
                "fuzzy": []
            },
            14: {
                "exact": ["kpi_container_threads", "kpi_container_file_descriptors"],
                "fuzzy": []
            }
        }
        self.offset = 20

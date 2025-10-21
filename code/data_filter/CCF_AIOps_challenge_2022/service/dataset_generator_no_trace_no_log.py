from shared_util.file_handler import FileHandler
from data_filter.CCF_AIOps_challenge_2022.base.base_generator import BaseGenerator
from data_filter.CCF_AIOps_challenge_2022.service.time_interval_label_generator import TimeIntervalLabelGenerator
from data_filter.CCF_AIOps_challenge_2022.service.metric_generator import MetricGenerator
from data_filter.CCF_AIOps_challenge_2022.service.trace_generator import TraceGenerator
from data_filter.CCF_AIOps_challenge_2022.service.log_generator import LogGenerator
from data_filter.CCF_AIOps_challenge_2022.service.ent_edge_index_generator import EntEdgeIndexGenerator
from data_filter.CCF_AIOps_challenge_2022.util.dataset_handler import DatasetHandler
from data_filter.CCF_AIOps_challenge_2022.service.api_generator import ApiGenerator


class DatasetGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()
        self.meta_data = {
            "modal_types": ['metric'],
            "ent_types": ['node', 'service', 'pod'],
            'ent_names': self.all_entity_list,
            'ent_type_index': {
                "node": (0, 6),
                "service": (6, 16),
                "pod": (16, 56)
            },
            'ent_features': {
                "metric": []
            },
            'max_ent_feature_num': {
                "node": {
                    "metric": 0
                },
                "service": {
                    "metric": 0
                },
                "pod": {
                    "metric": 0
                },
            },
            'o11y_names': {
                "metric": []
            },
            'o11y_length': {
                "metric": 0
            },
            'ent_fault_type_index': {
                "node": (0, 6),
                "service": (6, 15),
                "pod": (6, 15)
            },
            'ent_fault_type_weight': {
                "node": [138, 138, 138, 138, 138, 138],
                "service": [231, 231, 231, 231, 231, 231, 231, 231, 231],
                "pod": [183, 183, 183, 183, 183, 183, 183, 183, 183]
            },
            "fault_type_list": self.fault_type_list,
            "fault_type_related_o11y_names": self.fault_type_related_o11y_names
        }

    def get_base_data(self, modal_type_list, window_size):
        modal_dict = dict()

        self.meta_data['modal_types'] = modal_type_list
        if 'metric' in modal_type_list:
            temp = MetricGenerator().get_metric(window_size)
            self.meta_data['ent_features']['metric'] = temp['entity_features']
            self.meta_data['o11y_names']['metric'] = temp['metric_names']
            self.meta_data['o11y_length']['metric'] = len(temp['metric_names'])
            modal_dict['metric'] = temp['metric_data']

        for ent_type in self.meta_data['ent_types']:
            start_end_index_pair = self.meta_data['ent_type_index'][ent_type]
            for i in range(start_end_index_pair[0], start_end_index_pair[1]):
                for modal_type in modal_type_list:
                    ent_feature_num = self.meta_data['ent_features'][modal_type][i][1][1] - self.meta_data['ent_features'][modal_type][i][1][0]
                    self.meta_data['max_ent_feature_num'][ent_type][modal_type] = max(ent_feature_num, self.meta_data['max_ent_feature_num'][ent_type][modal_type])

        ent_edge_index = EntEdgeIndexGenerator().get_ent_edge_index(window_size)
        y = TimeIntervalLabelGenerator().get_time_interval_label(window_size)['y']

        return modal_dict, ent_edge_index, y

    def generate_rca_multimodal_dataset(self):
        modal_type_list = ['metric']
        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/merge_multimodal')
        for window_size in self.window_size_list:
            print("window_size: ", window_size)
            modal_dict, ent_edge_index, y = self.get_base_data(modal_type_list, window_size)

            DatasetHandler.split_and_save_dataset(modal_type_list=modal_type_list,
                                                  modal_data=modal_dict,
                                                  ent_edge_index=ent_edge_index,
                                                  valid_ratio=0.2,
                                                  y=y,
                                                  multi_class_label_format=True,
                                                  num_of_fault_types=15,
                                                  meta_data=self.meta_data,
                                                  save_file_path=f'{folder}/rca_multimodal_window_size_{window_size}_no_trace_no_log.pkl')


if __name__ == '__main__':
    dataset_generator = DatasetGenerator()
    dataset_generator.generate_rca_multimodal_dataset()

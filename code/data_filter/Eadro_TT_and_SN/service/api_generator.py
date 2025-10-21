from shared_util.file_handler import FileHandler
from data_filter.Eadro_TT_and_SN.base.base_generator import BaseGenerator
from data_filter.Eadro_TT_and_SN.base.base_sn_class import BaseSNClass
from data_filter.Eadro_TT_and_SN.base.base_tt_class import BaseTTClass
import numpy as np
import pickle


class ApiGenerator(BaseGenerator):
    def __init__(self, base):
        super().__init__(base)
        
    def from_entity_to_api(self, entity) -> list:
        api_list = []
        for api in self.base.all_api_list:
            entity_name = self.base.all_entity_list[entity]
            if api.startswith(f'{entity_name}:'):
                api_list.append(api)
        
        return list(set(api_list))
    
    def from_api_to_index(self, api) -> int:
        for i in range(len(self.base.all_api_list)):
            if api == self.base.all_api_list[i]:
                return i
        return -1

    def get_api_feature_info(self) -> (list, list):
        api_pattern_list = [f'cmdb_id: {api}' for api in self.base.all_api_list]
        feature_index, api_features, api_name_list = 0, [], []
        for i in range(len(api_pattern_list)):
            api_name_list.append(f'<intensity>; {api_pattern_list[i]}; type: upstream')
            api_name_list.append(f'<duration>; {api_pattern_list[i]}; type: upstream')
            api_name_list.append(f'<intensity>; {api_pattern_list[i]}; type: current')
            api_name_list.append(f'<duration>; {api_pattern_list[i]}; type: current')
            api_name_list.append(f'<intensity>; {api_pattern_list[i]}; type: downstream')
            api_name_list.append(f'<duration>; {api_pattern_list[i]}; type: downstream')
            api_features.append((i, (feature_index * 6, (feature_index + 1) * 6)))
            feature_index += 1
        return api_features, api_name_list

    def calculate_mean_and_std(self, data_dict):
        statistic_dict = dict()

        for i in range(len(self.base.all_api_list)):
            statistic_dict[self.base.all_api_list[i]] = {
                'train_valid_test': dict(),
                'train_valid': dict(),
                'normal': dict()
            }

        train_valid_test = np.concatenate([data_dict['train'], data_dict['valid'], data_dict['test']], axis=0)
        train_valid_test = train_valid_test.reshape(-1, train_valid_test.shape[2])
        for i in range(len(self.base.all_api_list)):
            statistic_dict[self.base.all_api_list[i]]['train_valid_test'] = {
                'mean': np.mean(train_valid_test, axis=0)[i * 6:(i + 1) * 6],
                'std': np.std(train_valid_test, axis=0)[i * 6:(i + 1) * 6],
                'max': np.max(train_valid_test, axis=0)[i * 6:(i + 1) * 6],
            }

        train_valid = np.concatenate([data_dict['train'], data_dict['valid']], axis=0)
        train_valid = train_valid.reshape(-1, train_valid.shape[2])
        for i in range(len(self.base.all_api_list)):
            statistic_dict[self.base.all_api_list[i]]['train_valid'] = {
                'mean': np.mean(train_valid, axis=0)[i * 6:(i + 1) * 6],
                'std': np.std(train_valid, axis=0)[i * 6:(i + 1) * 6],
                'max': np.max(train_valid, axis=0)[i * 6:(i + 1) * 6],
            }

        normal = np.concatenate([data_dict['z-score']], axis=0)
        normal = normal.reshape(-1, normal.shape[2])
        for i in range(len(self.base.all_api_list)):
            statistic_dict[self.base.all_api_list[i]]['normal'] = {
                'mean': np.mean(normal, axis=0)[i * 6:(i + 1) * 6],
                'std': np.std(normal, axis=0)[i * 6:(i + 1) * 6],
                'max': np.max(normal, axis=0)[i * 6:(i + 1) * 6],
            }

        return statistic_dict

    def generate_api_data(self):
        api_features, api_name_list = self.get_api_feature_info()
        for window_size in self.base.window_size_list:
            ground_truth_dict = self.ground_truth_dao.get_time_label_interval(window_size)
            raw_api_dict = self.raw_api_dao.load_api_csv()

            api_dict = dict()
            for dataset_type in ['train', 'valid', 'test', 'z-score']:
                api_dict[dataset_type] = []
                for ground_truth in ground_truth_dict[dataset_type]:
                    timestamp_list = raw_api_dict[ground_truth[0]].loc[:, 'timestamp'].tolist()
                    start_index = int((ground_truth[2] - timestamp_list[0]) / (timestamp_list[1] - timestamp_list[0]))
                    api_dict[dataset_type].append(raw_api_dict[ground_truth[0]].query(f'{timestamp_list[start_index]} <= timestamp < {timestamp_list[start_index + window_size]}').loc[:, api_name_list].values)
                api_dict[dataset_type] = np.array(api_dict[dataset_type])

            analysis_api_dict_before = {2: dict(), 3: dict()}
            for dataset_type in ['train', 'valid', 'test', 'z-score']:
                analysis_api_dict_before[2][dataset_type] = []
                analysis_api_dict_before[3][dataset_type] = []
                for i in range(len(ground_truth_dict[dataset_type])):
                    entity, fault_type = ground_truth_dict[dataset_type][i][4], ground_truth_dict[dataset_type][i][5]
                    if entity is not None and fault_type in analysis_api_dict_before.keys():
                        api_list = self.from_entity_to_api(entity)
                        for api in api_list:
                            if api not in self.base.all_api_list:
                                continue
                            api_index = self.from_api_to_index(api)
                            analysis_api_dict_before[fault_type][dataset_type].append((api_dict[dataset_type][i, :, entity * 6:(entity + 1) * 6], self.base.all_api_list[api_index]))

            api_dict = ApiGenerator.z_score_data(api_dict)
            statistic_dict = self.calculate_mean_and_std(api_dict)

            analysis_api_dict_after = {2: dict(), 3: dict()}
            for dataset_type in ['train', 'valid', 'test', 'z-score']:
                analysis_api_dict_after[2][dataset_type] = []
                analysis_api_dict_after[3][dataset_type] = []
                for i in range(len(ground_truth_dict[dataset_type])):
                    entity, fault_type = ground_truth_dict[dataset_type][i][4], ground_truth_dict[dataset_type][i][5]
                    if entity is not None and fault_type in analysis_api_dict_before.keys():
                        api_list = self.from_entity_to_api(entity)
                        for api in api_list:
                            if api not in self.base.all_api_list:
                                continue
                            api_index = self.from_api_to_index(api)                        
                            analysis_api_dict_after[fault_type][dataset_type].append((api_dict[dataset_type][i, :, entity * 6:(entity + 1) * 6], self.base.all_api_list[api_index], statistic_dict[self.base.all_api_list[api_index]]))

            with open(FileHandler.set_folder(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/dataset/api') + f'/api_window_size_{window_size}.pkl', 'wb') as f:
                pickle.dump({
                    'api_data': api_dict,
                    'entity_features': api_features,
                    'api_names': api_name_list
                }, f)

    def get_api(self, window_size) -> np:
        with open(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/dataset/api/api_window_size_{window_size}.pkl', 'rb') as f:
            api = pickle.load(f)
            return api


if __name__ == '__main__':
    api_generator = ApiGenerator(BaseTTClass())
    api_generator.generate_api_data()

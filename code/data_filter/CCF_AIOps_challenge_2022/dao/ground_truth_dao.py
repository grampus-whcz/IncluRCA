import pandas as pd
import glob
import os

from data_filter.CCF_AIOps_challenge_2022.base.base_class import BaseClass


class GroundTruthDao(BaseClass):
    def __init__(self):
        super().__init__()

    def get_ground_truth(self, dataset_type: str):
        result_dict = dict()

        data_base_path = f'{self.config.data_dict["ground_truth"][dataset_type]}'
        print(data_base_path)

        file_list = glob.glob(f'{data_base_path}/*.csv')
        
        desired_order_test = ["05-01", "05-07", "05-09", "05-03", "05-05"]

        # --- 定义排序键函数 ---
        def sort_key1(file_path):
            filename = os.path.basename(file_path)
            # 提取日期部分，例如 '2022-05-01' -> '05-01'
            date_part = filename.replace('groundtruth-', '').replace('.csv', '').split('2022-')[-1]
            try:
                # 返回 desired_order 中 date_part 的索引
                # 索引越小，排序越靠前
                return desired_order_test.index(date_part)
            except ValueError:
                # 如果文件名中的日期不在 desired_order 中，可以选择将其放在最后
                # 或者抛出错误，取决于你的需求
                return len(desired_order_test) # 将其放在最后
            
        desired_order_train_valid = ["3-2022-03-24", "2-2022-03-21", "1-2022-03-21", "3-2022-03-21", "3-2022-03-20", "1-2022-03-20", "2-2022-03-20"]

        # --- 定义排序键函数 ---
        def sort_key2(file_path):
            filename = os.path.basename(file_path)
            # 提取日期部分，例如 '2022-05-01' -> '05-01'
            date_part = filename.replace('groundtruth-', '').replace('.csv', '').split('k8s-')[-1]
            try:
                # 返回 desired_order 中 date_part 的索引
                # 索引越小，排序越靠前
                return desired_order_train_valid.index(date_part)
            except ValueError:
                # 如果文件名中的日期不在 desired_order 中，可以选择将其放在最后
                # 或者抛出错误，取决于你的需求
                return len(desired_order_train_valid) # 将其放在最后

        # --- 使用自定义键进行排序 ---
        if dataset_type == 'train_valid':
            file_list = sorted(file_list, key=sort_key2)
        else:
            file_list = sorted(file_list, key=sort_key1)
        
        for file in file_list:
            if dataset_type == 'train_valid':
                date = '2022-' + file.split('/')[-1].replace('.csv', '').split('2022-')[-1]
                cloud_bed = 'cloudbed-' + file.split('/')[-1].replace('.csv', '').split('-2022')[0].split('-')[-1]
            else:
                date = '2022-' + file.split('/')[-1].replace('.csv', '').split('2022-')[-1]
                cloud_bed = 'cloudbed'
            if date not in result_dict.keys():
                result_dict[date] = dict()
            ground_truth_df = pd.read_csv(file)
            result_dict[date][cloud_bed] = ground_truth_df.to_dict('list')
        return result_dict

    def analyze_ground_truth(self):
        def count_ground_truth_type(ground_truth_dict):
            result_dict = {
                'fault_type_count': dict(),
                'cmdb_count': dict(),
                'all_count': dict()
            }

            for date in ground_truth_dict.keys():
                for cloud_bed in ground_truth_dict[date].keys():
                    for i in range(len(ground_truth_dict[date][cloud_bed]['timestamp'])):
                        fault_type = ground_truth_dict[date][cloud_bed]['failure_type'][i]
                        cmdb_id = ground_truth_dict[date][cloud_bed]['cmdb_id'][i]
                        level = ground_truth_dict[date][cloud_bed]['level'][i]

                        if fault_type not in result_dict['fault_type_count'].keys():
                            result_dict['fault_type_count'][fault_type] = 0
                        result_dict['fault_type_count'][fault_type] += 1

                        if level == 'pod':
                            cmdb_id = f'pod/{cmdb_id.replace("2-0", "").replace("-0", "").replace("-1", "").replace("-2", "")}'
                            count = 1
                        elif level == 'node':
                            cmdb_id = f'node/{cmdb_id[0:4]}'
                            count = 1
                        else:
                            cmdb_id = f'pod/{cmdb_id}'
                            count = 4
                        if cmdb_id not in result_dict['cmdb_count'].keys():
                            result_dict['cmdb_count'][cmdb_id] = 0
                        result_dict['cmdb_count'][cmdb_id] += count

                        all_info = f'{cmdb_id}/{fault_type}'
                        if all_info not in result_dict['all_count'].keys():
                            result_dict['all_count'][all_info] = 0
                        result_dict['all_count'][all_info] += count

            return result_dict

        train_valid_ground_truth = self.get_ground_truth('train_valid')
        test_ground_truth = self.get_ground_truth('test')

        train_valid_result_dict = count_ground_truth_type(train_valid_ground_truth)
        test_result_dict = count_ground_truth_type(test_ground_truth)

        not_seen_fault = set(test_result_dict['all_count'].keys()) - set(train_valid_result_dict['all_count'].keys())
        seen_and_not_happened_fault = set(train_valid_result_dict['all_count'].keys()) - set(test_result_dict['all_count'].keys())
        ...


if __name__ == '__main__':
    ground_truth_dao = GroundTruthDao()
    ground_truth_dao.get_ground_truth('train_valid')
    ground_truth_dao.analyze_ground_truth()

# from sklearn.model_selection import train_test_split
# import numpy as np
# import pickle
# import random


# random.seed(409)


# class DatasetHandler:
#     @staticmethod
#     def split_and_save_dataset(modal_type_list: list,
#                                modal_data: dict,
#                                ent_edge_index: dict,
#                                valid_ratio: float,
#                                y: dict,
#                                multi_class_label_format: bool,
#                                num_of_fault_types: int,
#                                meta_data: dict,
#                                save_file_path: str):
#         result_dict = {
#             'data': dict(),
#             'meta_data': meta_data
#         }

#         test_size = valid_ratio
#         random_state = 409

#         train_valid_list, test_list = [], []
#         for modal_type in modal_type_list:
#             train_valid_list.append(modal_data[modal_type]['train_valid'])
#             test_list.append(modal_data[modal_type]['test'])
#         train_valid_list.append(ent_edge_index['train_valid'])
#         test_list.append(ent_edge_index['test'])
#         train_valid_list.append(y['train_valid'])
#         test_list.append(y['test'])

#         train_valid = train_test_split(*tuple(train_valid_list),
#                                        test_size=test_size,
#                                        random_state=random_state,
#                                        shuffle=True)

#         data = dict()
#         data['train'], data['valid'], data['test'] = train_valid[::2], train_valid[1::2], test_list
#         # print("####  size: ", len(data['train'][0]))
#         data['train'] = DatasetHandler.over_sampling(data['train'])
#         # print("####  size: ", len(data['train'][0]))
        
#         ## =========== test whether there are abnormal data =================
#         for i, modal_name in enumerate(modal_type_list):
#             print(f"\n=== Modal {modal_name} (index {i}) ===")
#             shapes = []
#             for sample in data['train'][i]:
#                 if hasattr(sample, 'shape'):
#                     shapes.append(sample.shape)
#                 else:
#                     shapes.append(('not array', type(sample)))
            
#             # 统计 unique shapes
#             from collections import Counter
#             shape_counts = Counter(shapes)
#             print("Unique shapes in train:", shape_counts)

#             # 对比 valid
#             valid_shapes = [s.shape if hasattr(s, 'shape') else ('not array', type(s)) for s in data['valid'][i]]
#             valid_shape_counts = Counter(valid_shapes)
#             print("Unique shapes in valid:", valid_shape_counts)
            
#         ## =================================
        
#         ## for those with more than two shapes, please run the following code.
#         ## 
#         # def align_sequence_length_0(samples, target_len=17):
#         #     aligned = []
#         #     for s in samples:
#         #         if s.shape[0] > target_len:
#         #             # 截断（保留前 target_len 步）
#         #             aligned.append(s[:target_len])
#         #         elif s.shape[0] < target_len:
#         #             # 填充（用0）
#         #             pad_width = ((0, target_len - s.shape[0]), (0, 0))
#         #             padded = np.pad(s, pad_width, mode='constant', constant_values=0)
#         #             aligned.append(padded)
#         #         else:
#         #             aligned.append(s)
#         #     return aligned
        
#         # import numpy as np

#         # def align_sequence_length_1(samples, target_len=17):
#         #     aligned = []
#         #     for s in samples:
#         #         current_len = s.shape[0]
#         #         if current_len > target_len:
#         #             # 截断：保留前 target_len 步
#         #             aligned.append(s[:target_len])
#         #         elif current_len < target_len:
#         #             # 填充：用最后一帧重复补齐
#         #             last_frame = s[-1:]  # 保持维度 (1, F)
#         #             repeat_times = target_len - current_len
#         #             padded = np.concatenate([s, np.repeat(last_frame, repeat_times, axis=0)], axis=0)
#         #             aligned.append(padded)
#         #         else:
#         #             # 长度正好，直接保留
#         #             aligned.append(s)
#         #     return aligned
        
#         # for data_type in ['train', 'valid', 'test']:
#         #     for i in range(len(modal_type_list)):
#         #         data[data_type][i] = align_sequence_length_1(data[data_type][i], target_len=17)

#         for data_type in ['train', 'valid', 'test']:
#             for i in range(len(modal_type_list)):
#                 # print(f"i: {i}")
#                 # print(f'x_{modal_type_list[i]}')
#                 # print(len(data[data_type][i]))
#                 # print(len(data[data_type][i][0][0]))
#                 # data_tmp = np.array(data[data_type][i], dtype=object)
#                 # print(data_tmp.shape)
#                 # result_dict['data'][f'x_{modal_type_list[i]}_{data_type}'] = data_tmp
#                 # result_dict['data'][f'x_{modal_type_list[i]}_{data_type}'] = np.array(data[data_type][i], dtype=object) # old
#                 try:
#                     arr = np.stack(data[data_type][i])  # 要求所有样本 shape 一致
#                 except ValueError as e:
#                     print(f"Shape mismatch in {data_type}, modal {i}: {e}")
#                     # fallback to object array or debug
#                     arr = np.array(data[data_type][i], dtype=object)
#                 result_dict['data'][f'x_{modal_type_list[i]}_{data_type}'] = arr
#             result_dict['data'][f'ent_edge_index_{data_type}'] = data[data_type][len(modal_type_list)]

#             if multi_class_label_format:
#                 data[data_type][len(modal_type_list) + 1] = DatasetHandler.label_to_multi_class_format(data[data_type][len(modal_type_list) + 1], num_of_fault_types)

#             result_dict['data'][f'y_{data_type}'] = np.array(data[data_type][len(modal_type_list) + 1])

#         with open(save_file_path, 'wb') as f:
#             pickle.dump(result_dict, f, protocol=4)

#     @staticmethod
#     def label_to_multi_class_format(raw_y, num_of_fault_types=15):
#         y = []
#         raw_y = np.array(raw_y)
#         for i in range(raw_y.shape[0]):
#             y.append([])
#             for j in range(raw_y.shape[1]):
#                 y[-1].append(np.zeros(num_of_fault_types))
#                 if raw_y[i][j] != 0:
#                     y[-1][-1][int(raw_y[i][j] - 1)] = 1
#         return np.array(y)

#     @staticmethod
#     def over_sampling(train):
#         x = np.array(train, dtype=object).transpose().tolist()
#         y_train = train[-1]

#         temp_y = []
#         fault_type_list = []
#         for label in y_train:
#             fault_type = 'None'
#             if np.count_nonzero(label) > 1:
#                 fault_type = f'service:{int(np.max(label))}'
#             elif np.count_nonzero(label) == 1:
#                 fault_type = f'{int(np.max(label))}'
#             if fault_type not in fault_type_list:
#                 fault_type_list.append(fault_type)
#             temp_y.append(fault_type_list.index(fault_type))
#         x, temp_y = np.array(x, dtype=object), np.array(temp_y, dtype=object)

#         k = 0
#         index_dict = dict()
#         for i in range(len(fault_type_list)):
#             index_dict[i] = np.where(temp_y == i)[0]
#             k = max(k, index_dict[i].shape[0])
#         for i in range(len(fault_type_list)):
#             index_dict[i] = random.choices(population=index_dict[i].tolist(), k=k)

#         indices = []
#         for i in range(len(fault_type_list)):
#             indices.extend(index_dict[i])
#         random.shuffle(indices)
#         indices = np.array(indices)

#         x = x[indices]
#         return np.array(x).transpose().tolist()


# from sklearn.model_selection import train_test_split
# import numpy as np
# import pickle
# import random

# random.seed(409)

# class DatasetHandler:
#     @staticmethod
#     def split_and_save_dataset(modal_type_list: list,
#                                modal_data: dict,
#                                ent_edge_index: dict,
#                                valid_ratio: float,
#                                y: dict,
#                                multi_class_label_format: bool,
#                                num_of_fault_types: int,
#                                meta_data: dict,
#                                save_file_path: str):
#         result_dict = {
#             'data': dict(),
#             'meta_data': meta_data
#         }

#         test_size = valid_ratio
#         random_state = 409

#         train_valid_list, test_list = [], []
#         normal_train_valid_list = []
#         normal_test_list = []

#         # 1. 构造故障数据和normal数据列表（从modal_data提取）
#         for modal_type in modal_type_list:
#             # 故障数据
#             train_valid_data = modal_data[modal_type]['train_valid']
#             test_data = modal_data[modal_type]['test']
#             train_valid_list.append(train_valid_data)
#             test_list.append(test_data)
#             # normal数据
#             normal_train_valid_data = modal_data[modal_type]['normal_for_train_valid']
#             normal_test_data = modal_data[modal_type]['normal_for_test']
#             normal_train_valid_list.append(normal_train_valid_data)
#             normal_test_list.append(normal_test_data)

#         # 记录基准样本数（以故障数据的第一个元素为准，确保一致性）
#         base_sample_num = len(train_valid_list[0]) if train_valid_list else 0

#         # 添加实体边索引（修复核心：填充空列表为与基准样本数匹配的占位数据）
#         train_valid_list.append(ent_edge_index['train_valid'])
#         test_list.append(ent_edge_index['test'])
#         # 处理normal实体边索引：若为空，创建与基准样本数一致的占位列表
#         normal_ent_edge_train_valid = ent_edge_index.get('normal_for_train_valid', [])
#         if len(normal_ent_edge_train_valid) == 0 and base_sample_num > 0:
#             # 占位数据：可根据ent_edge_index['train_valid']的元素格式调整，这里用空列表占位
#             normal_ent_edge_train_valid = [[] for _ in range(base_sample_num)]
#         normal_train_valid_list.append(normal_ent_edge_train_valid)
#         # 处理normal测试集实体边索引
#         normal_ent_edge_test = ent_edge_index.get('normal_for_test', [])
#         test_base_sample_num = len(test_list[0]) if test_list else 0
#         if len(normal_ent_edge_test) == 0 and test_base_sample_num > 0:
#             normal_ent_edge_test = [[] for _ in range(test_base_sample_num)]
#         normal_test_list.append(normal_ent_edge_test)

#         # 添加标签
#         train_valid_list.append(y['train_valid'])
#         test_list.append(y['test'])
#         # 处理normal标签：若为空，创建与基准样本数一致的占位列表
#         normal_y_train_valid = y.get('normal_for_train_valid', [])
#         if len(normal_y_train_valid) == 0 and base_sample_num > 0:
#             # 占位数据：与y['train_valid']格式一致（这里假设y是二维结构，可根据实际调整）
#             normal_y_train_valid = [np.zeros_like(y['train_valid'][0]) for _ in range(base_sample_num)]
#         normal_train_valid_list.append(normal_y_train_valid)
#         # 处理normal测试集标签
#         normal_y_test = y.get('normal_for_test', [])
#         if len(normal_y_test) == 0 and test_base_sample_num > 0:
#             normal_y_test = [np.zeros_like(y['test'][0]) for _ in range(test_base_sample_num)]
#         normal_test_list.append(normal_y_test)

#         # 2. 验证并过滤不一致的元素（双重保障，避免报错）
#         def filter_consistent_data(data_list, base_num):
#             """过滤数据列表，确保所有元素样本数与基准数一致"""
#             filtered_list = []
#             for item in data_list:
#                 if len(item) == base_num:
#                     filtered_list.append(item)
#                 else:
#                     # 若不一致，创建占位数据补充
#                     placeholder = [[] for _ in range(base_num)]
#                     filtered_list.append(placeholder)
#             return filtered_list

#         # 过滤normal_train_valid_list（以base_sample_num为基准）
#         if base_sample_num > 0:
#             normal_train_valid_list = filter_consistent_data(normal_train_valid_list, base_sample_num)
#         # 过滤normal_test_list（以test_base_sample_num为基准）
#         if test_base_sample_num > 0:
#             normal_test_list = filter_consistent_data(normal_test_list, test_base_sample_num)

#         # 3. 划分训练集和验证集（故障数据）
#         train_valid = train_test_split(*tuple(train_valid_list),
#                                        test_size=test_size,
#                                        random_state=random_state,
#                                        shuffle=True)

#         # 构造故障数据字典
#         data = dict()
#         data['train'], data['valid'], data['test'] = train_valid[::2], train_valid[1::2], test_list
        
#         # 4. 对normal数据执行相同划分（此时数据已一致，不会报错）
#         normal_train_valid = train_test_split(*tuple(normal_train_valid_list),
#                                               test_size=test_size,
#                                               random_state=random_state,
#                                               shuffle=True)
#         normal_data_dict = dict()
#         normal_data_dict['train'] = normal_train_valid[::2]
#         normal_data_dict['valid'] = normal_train_valid[1::2]
#         normal_data_dict['test'] = normal_test_list

#         # 5. 过采样train集，并获取过采样索引
#         data['train'], oversample_indices = DatasetHandler.over_sampling(data['train'])

#         # 6. 复用采样索引处理normal train集
#         normal_train = np.array(normal_data_dict['train'], dtype=object).transpose().tolist()
#         normal_train = np.array(normal_train, dtype=object)
#         normal_train_oversampled = normal_train[oversample_indices]
#         normal_train_oversampled = np.array(normal_train_oversampled).transpose().tolist()
#         normal_data_dict['train'] = normal_train_oversampled

#         ## =========== 原有数据校验逻辑（不变） =================
#         for i, modal_name in enumerate(modal_type_list):
#             print(f"\n=== Modal {modal_name} (index {i}) ===")
#             shapes = []
#             for sample in data['train'][i]:
#                 if hasattr(sample, 'shape'):
#                     shapes.append(sample.shape)
#                 else:
#                     shapes.append(('not array', type(sample)))
            
#             from collections import Counter
#             shape_counts = Counter(shapes)
#             print("Unique shapes in train:", shape_counts)

#             valid_shapes = [s.shape if hasattr(s, 'shape') else ('not array', type(s)) for s in data['valid'][i]]
#             valid_shape_counts = Counter(valid_shapes)
#             print("Unique shapes in valid:", valid_shape_counts)
#         ## =================================

#         ## =========== 数据保存逻辑（不变） =================
#         for data_type in ['train', 'valid', 'test']:
#             # 保存故障数据
#             for i in range(len(modal_type_list)):
#                 try:
#                     arr = np.stack(data[data_type][i])
#                 except ValueError as e:
#                     print(f"Shape mismatch in {data_type}, modal {i}: {e}")
#                     arr = np.array(data[data_type][i], dtype=object)
#                 result_dict['data'][f'x_{modal_type_list[i]}_{data_type}'] = arr
#             result_dict['data'][f'ent_edge_index_{data_type}'] = data[data_type][len(modal_type_list)]

#             if multi_class_label_format:
#                 data[data_type][len(modal_type_list) + 1] = DatasetHandler.label_to_multi_class_format(data[data_type][len(modal_type_list) + 1], num_of_fault_types)

#             result_dict['data'][f'y_{data_type}'] = np.array(data[data_type][len(modal_type_list) + 1])

#             # 保存normal数据
#             for i in range(len(modal_type_list)):
#                 try:
#                     normal_arr = np.stack(normal_data_dict[data_type][i])
#                 except ValueError as e:
#                     print(f"Shape mismatch in normal {data_type}, modal {i}: {e}")
#                     normal_arr = np.array(normal_data_dict[data_type][i], dtype=object)
#                 result_dict['data'][f'x_{modal_type_list[i]}_{data_type}_normal'] = normal_arr
#             result_dict['data'][f'ent_edge_index_{data_type}_normal'] = normal_data_dict[data_type][len(modal_type_list)]
#             if multi_class_label_format and len(normal_data_dict[data_type]) > len(modal_type_list) + 1:
#                 normal_data_dict[data_type][len(modal_type_list) + 1] = DatasetHandler.label_to_multi_class_format(normal_data_dict[data_type][len(modal_type_list) + 1], num_of_fault_types)
#             result_dict['data'][f'y_{data_type}_normal'] = np.array(normal_data_dict[data_type][len(modal_type_list) + 1])

#         with open(save_file_path, 'wb') as f:
#             pickle.dump(result_dict, f, protocol=4)

#     @staticmethod
#     def over_sampling(train):
#         x = np.array(train, dtype=object).transpose().tolist()
#         y_train = train[-1]

#         temp_y = []
#         fault_type_list = []
#         for label in y_train:
#             fault_type = 'None'
#             if np.count_nonzero(label) > 1:
#                 fault_type = f'service:{int(np.max(label))}'
#             elif np.count_nonzero(label) == 1:
#                 fault_type = f'{int(np.max(label))}'
#             if fault_type not in fault_type_list:
#                 fault_type_list.append(fault_type)
#             temp_y.append(fault_type_list.index(fault_type))
#         x, temp_y = np.array(x, dtype=object), np.array(temp_y, dtype=object)

#         k = 0
#         index_dict = dict()
#         for i in range(len(fault_type_list)):
#             index_dict[i] = np.where(temp_y == i)[0]
#             k = max(k, index_dict[i].shape[0])
#         for i in range(len(fault_type_list)):
#             index_dict[i] = random.choices(population=index_dict[i].tolist(), k=k)

#         indices = []
#         for i in range(len(fault_type_list)):
#             indices.extend(index_dict[i])
#         random.shuffle(indices)
#         indices = np.array(indices)

#         x = x[indices]
#         train_oversampled = np.array(x).transpose().tolist()
#         return train_oversampled, indices

#     @staticmethod
#     def label_to_multi_class_format(raw_y, num_of_fault_types=15):
#         y = []
#         raw_y = np.array(raw_y)
#         for i in range(raw_y.shape[0]):
#             y.append([])
#             for j in range(raw_y.shape[1]):
#                 y[-1].append(np.zeros(num_of_fault_types))
#                 if raw_y[i][j] != 0:
#                     y[-1][-1][int(raw_y[i][j] - 1)] = 1
#         return np.array(y)


from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import random

# --- 固定随机种子 ---
random.seed(409)
np.random.seed(409)

class DatasetHandler:
    @staticmethod
    def split_and_save_dataset(modal_type_list: list,
                               modal_data: dict,
                               ent_edge_index: dict,
                               valid_ratio: float, # 例如 0.1
                               y: dict,
                               multi_class_label_format: bool,
                               num_of_fault_types: int,
                               meta_data: dict,
                               save_file_path: str):
        result_dict = {
            'data': dict(),
            'meta_data': meta_data
        }

        test_size = valid_ratio
        random_state = 409

        # --- 1. 提取 fault 和 normal 的 train_valid/test 数据 ---
        fault_train_valid_list, fault_test_list = [], []
        normal_train_valid_list, normal_test_list = [], []

        for modal_type in modal_type_list:
            fault_train_valid_list.append(modal_data[modal_type]['train_valid'])
            fault_test_list.append(modal_data[modal_type]['test'])
        fault_train_valid_list.append(ent_edge_index['train_valid'])
        fault_test_list.append(ent_edge_index['test'])
        fault_train_valid_list.append(y['train_valid'])
        fault_test_list.append(y['test'])

        for modal_type in modal_type_list:
            normal_train_valid_list.append(modal_data[modal_type]['normal_for_train_valid'])
            normal_test_list.append(modal_data[modal_type]['normal_for_test'])
        normal_train_valid_list.append(ent_edge_index['normal_for_train_valid'])
        normal_test_list.append(ent_edge_index['normal_for_test'])
        normal_train_valid_list.append(y['normal_for_train_valid'])
        normal_test_list.append(y['normal_for_test'])

        # 验证长度是否一致
        assert len(fault_train_valid_list[0]) == len(normal_train_valid_list[0]), \
            f"Mismatch in train_valid lengths: fault={len(fault_train_valid_list[0])}, normal={len(normal_train_valid_list[0])}"
        assert len(fault_test_list[0]) == len(normal_test_list[0]), \
            f"Mismatch in test lengths: fault={len(fault_test_list[0])}, normal={len(normal_test_list[0])}"

        original_train_valid_size = len(fault_train_valid_list[0])

        # --- 2. 按元素合并 fault 和 normal 数据 (创建新的复合样本) ---
        combined_train_valid_list = []
        combined_test_list = []

        # 合并 train_valid
        for i in range(len(fault_train_valid_list)):
            combined_component = []
            for j in range(original_train_valid_size):
                # 将 fault 和 normal 的第 j 个样本合并成一个元组
                combined_component.append((fault_train_valid_list[i][j], normal_train_valid_list[i][j]))
            combined_train_valid_list.append(combined_component)

        # 合并 test
        for i in range(len(fault_test_list)):
            combined_component = []
            for j in range(len(fault_test_list[i])): # 使用 test 的长度
                combined_component.append((fault_test_list[i][j], normal_test_list[i][j]))
            combined_test_list.append(combined_component)

        # --- 3. 对合并后的复合数据集进行划分 ---
        combined_train_valid_split = train_test_split(*tuple(combined_train_valid_list),
                                                      test_size=test_size,
                                                      random_state=random_state,
                                                      shuffle=True)
        combined_data = dict()
        combined_data['train'] = combined_train_valid_split[::2] # [train_part0, train_part1, ...]
        combined_data['valid'] = combined_train_valid_split[1::2] # [valid_part0, valid_part1, ...]
        combined_data['test'] = combined_test_list # Test 数据不变

        # --- 4. 对合并后的训练集进行过采样 ---
        # over_sampling_combined 仅基于 fault_sample 的标签进行重采样决策
        # 但重采样索引同步应用于 (fault_sample, normal_sample) 对
        combined_data['train'], _ = DatasetHandler.over_sampling_combined(combined_data['train'])

        # --- 5. 分离 fault 和 normal 数据 ---
        # 将复合样本 (fault_sample, normal_sample) 分离回 fault 和 normal 列表
        fault_data = {'train': [], 'valid': [], 'test': []}
        normal_data = {'train': [], 'valid': [], 'test': []}

        for data_type in ['train', 'valid', 'test']:
            for i in range(len(combined_data[data_type])):
                fault_component = []
                normal_component = []
                for combined_sample in combined_data[data_type][i]:
                    # 假设 combined_sample 是一个元组 (fault_sample, normal_sample)
                    fault_sample, normal_sample = combined_sample
                    fault_component.append(fault_sample)
                    normal_component.append(normal_sample)
                fault_data[data_type].append(fault_component)
                normal_data[data_type].append(normal_component)

        # --- 6. 原有的数据校验逻辑 (针对 fault 数据) ---
        for i, modal_name in enumerate(modal_type_list):
            print(f"\n=== Modal {modal_name} (index {i}) ===")
            shapes = []
            for sample in fault_data['train'][i]:
                if hasattr(sample, 'shape'):
                    shapes.append(sample.shape)
                else:
                    shapes.append(('not array', type(sample)))
            from collections import Counter
            shape_counts = Counter(shapes)
            print("Unique shapes in fault train:", shape_counts)

            valid_shapes = [s.shape if hasattr(s, 'shape') else ('not array', type(s)) for s in fault_data['valid'][i]]
            valid_shape_counts = Counter(valid_shapes)
            print("Unique shapes in fault valid:", valid_shape_counts)

        # --- 7. 原有的数据保存逻辑 ---
        for data_type in ['train', 'valid', 'test']:
            # 保存 fault 数据
            for i in range(len(modal_type_list)):
                try:
                    arr = np.stack(fault_data[data_type][i])
                except ValueError as e:
                    print(f"Shape mismatch in fault {data_type}, modal {i}: {e}")
                    arr = np.array(fault_data[data_type][i], dtype=object)
                result_dict['data'][f'x_{modal_type_list[i]}_{data_type}'] = arr
            result_dict['data'][f'ent_edge_index_{data_type}'] = fault_data[data_type][len(modal_type_list)]

            if multi_class_label_format:
                fault_data[data_type][len(modal_type_list) + 1] = DatasetHandler.label_to_multi_class_format(fault_data[data_type][len(modal_type_list) + 1], num_of_fault_types)

            result_dict['data'][f'y_{data_type}'] = np.array(fault_data[data_type][len(modal_type_list) + 1])

            # 保存 normal 数据
            for i in range(len(modal_type_list)):
                try:
                    normal_arr = np.stack(normal_data[data_type][i])
                except ValueError as e:
                    print(f"Shape mismatch in normal {data_type}, modal {i}: {e}")
                    normal_arr = np.array(normal_data[data_type][i], dtype=object)
                result_dict['data'][f'x_{modal_type_list[i]}_{data_type}_normal'] = normal_arr
            result_dict['data'][f'ent_edge_index_{data_type}_normal'] = normal_data[data_type][len(modal_type_list)]
            if multi_class_label_format and len(normal_data[data_type]) > len(modal_type_list) + 1:
                normal_data[data_type][len(modal_type_list) + 1] = DatasetHandler.label_to_multi_class_format(normal_data[data_type][len(modal_type_list) + 1], num_of_fault_types)
            result_dict['data'][f'y_{data_type}_normal'] = np.array(normal_data[data_type][len(modal_type_list) + 1])

        with open(save_file_path, 'wb') as f:
            pickle.dump(result_dict, f, protocol=4)

    @staticmethod
    def label_to_multi_class_format(raw_y, num_of_fault_types=15):
        y = []
        raw_y = np.array(raw_y)
        for i in range(raw_y.shape[0]):
            y.append([])
            for j in range(raw_y.shape[1]):
                y[-1].append(np.zeros(num_of_fault_types))
                if raw_y[i][j] != 0:
                    y[-1][-1][int(raw_y[i][j] - 1)] = 1
        return np.array(y)

    @staticmethod
    def over_sampling_combined(combined_train_list):
        """
        Modified over_sampling that works on combined (fault, normal) samples.
        It uses the fault sample's label (the first element of the label pair) to determine the sampling strategy.
        The resulting sampling indices are then applied to ALL components of the combined list,
        including the normal parts, ensuring synchronization.
        """
        # combined_train_list format: [ [ (f0, n0), (f1, n1), ... ], [ (e0, o0), (e1, o1), ... ], [ (l0, m0), (l1, m1), ... ] ]
        # where f, e, l are fault components (modal, edge, label)
        # and n, o, m are corresponding normal components
        # The last component contains the label pairs: [ (label_fault_0, label_normal_0), ... ]

        # Extract fault labels from the last component's pairs for sampling strategy
        y_combined = combined_train_list[-1]
        y_fault = [pair[0] for pair in y_combined] # Extract fault labels only

        temp_y = []
        fault_type_list = []
        for label in y_fault:
            fault_type = 'None'
            if np.count_nonzero(label) > 1:
                fault_type = f'service:{int(np.max(label))}'
            elif np.count_nonzero(label) == 1:
                fault_type = f'{int(np.max(label))}'
            if fault_type not in fault_type_list:
                fault_type_list.append(fault_type)
            temp_y.append(fault_type_list.index(fault_type))

        temp_y = np.array(temp_y, dtype=object)

        k = 0
        index_dict = dict()
        for i in range(len(fault_type_list)):
            indices = np.where(temp_y == i)[0]
            index_dict[i] = indices.tolist()
            k = max(k, indices.shape[0])
        for i in range(len(fault_type_list)):
            index_dict[i] = random.choices(population=index_dict[i], k=k)

        indices = []
        for i in range(len(fault_type_list)):
            indices.extend(index_dict[i])
        random.shuffle(indices) # Uses global random state
        indices = np.array(indices)

        # Apply the same indices (determined by fault labels) to ALL components
        # This ensures that (fault_sample, normal_sample) pairs stay together after resampling
        combined_train_oversampled = []
        for component_list in combined_train_list:
            oversampled_component = [component_list[idx] for idx in indices]
            combined_train_oversampled.append(oversampled_component)

        return combined_train_oversampled, indices

# 这个方案现在正确地实现了：
# 1. 将 fault 和 normal 对应样本合并 (f_i, n_i)
# 2. 对合并后的数据进行划分
# 3. 重采样时，仅依据 f_i 的标签决定采样策略
# 4. 将由此策略确定的采样索引，同步应用到 (f_i, n_i) 对以及所有其他模态/边/标签上
# 5. 最后将 (f_i, n_i) 对分离回 fault 和 normal 列表
# 这确保了划分和重采样与 fault 数据同步，并且重采样逻辑仅由 fault 部分驱动。


# from sklearn.model_selection import train_test_split
# import numpy as np
# import pickle
# import random


# random.seed(409)


# class DatasetHandler:
#     @staticmethod
#     def split_and_save_dataset(modal_type_list: list,
#                                modal_data: dict,
#                                ent_edge_index: dict,
#                                valid_ratio: float,
#                                y: dict,
#                                multi_class_label_format: bool,
#                                num_of_fault_types: int,
#                                meta_data: dict,
#                                save_file_path: str):
#         result_dict = {
#             'data': dict(),
#             'meta_data': meta_data
#         }

#         test_size = valid_ratio
#         random_state = 409

#         train_valid_list, test_list = [], []
#         for modal_type in modal_type_list:
#             train_valid_list.append(modal_data[modal_type]['train_valid'])
#             test_list.append(modal_data[modal_type]['test'])
#         train_valid_list.append(ent_edge_index['train_valid'])
#         test_list.append(ent_edge_index['test'])
#         train_valid_list.append(y['train_valid'])
#         test_list.append(y['test'])

#         train_valid = train_test_split(*tuple(train_valid_list),
#                                        test_size=test_size,
#                                        random_state=random_state,
#                                        shuffle=True)

#         data = dict()
#         data['train'], data['valid'], data['test'] = train_valid[::2], train_valid[1::2], test_list
        
#         print("####  size: ", len(data['train'][0]))
#         data['train'] = DatasetHandler.over_sampling(data['train'])
#         print("####  size: ", len(data['train'][0]))

#         for data_type in ['train', 'valid', 'test']:
#             for i in range(len(modal_type_list)):
#                 inner_shapes = [np.array(item).shape for item in data[data_type][i]]
#                 print("Inner shapes:", inner_shapes)
#                 print("Unique shapes:", set(inner_shapes))
#                 result_dict['data'][f'x_{modal_type_list[i]}_{data_type}'] = np.array(data[data_type][i], dtype=object)
#             result_dict['data'][f'ent_edge_index_{data_type}'] = data[data_type][len(modal_type_list)]

#             if multi_class_label_format:
#                 data[data_type][len(modal_type_list) + 1] = DatasetHandler.label_to_multi_class_format(data[data_type][len(modal_type_list) + 1], num_of_fault_types)

#             result_dict['data'][f'y_{data_type}'] = np.array(data[data_type][len(modal_type_list) + 1])

#         with open(save_file_path, 'wb') as f:
#             pickle.dump(result_dict, f, protocol=4)

#     @staticmethod
#     def label_to_multi_class_format(raw_y, num_of_fault_types=15):
#         y = []
#         raw_y = np.array(raw_y)
#         for i in range(raw_y.shape[0]):
#             y.append([])
#             for j in range(raw_y.shape[1]):
#                 y[-1].append(np.zeros(num_of_fault_types))
#                 if raw_y[i][j] != 0:
#                     y[-1][-1][int(raw_y[i][j] - 1)] = 1
#         return np.array(y)

#     @staticmethod
#     def over_sampling(train):
#         x = np.array(train, dtype=object).transpose().tolist()
#         y_train = train[-1]

#         temp_y = []
#         fault_type_list = []
#         for label in y_train:
#             fault_type = 'None'
#             if np.count_nonzero(label) > 1:
#                 fault_type = f'service:{int(np.max(label))}'
#             elif np.count_nonzero(label) == 1:
#                 fault_type = f'{int(np.max(label))}'
#             if fault_type not in fault_type_list:
#                 fault_type_list.append(fault_type)
#             temp_y.append(fault_type_list.index(fault_type))
#         x, temp_y = np.array(x, dtype=object), np.array(temp_y, dtype=object)

#         k = 0
#         index_dict = dict()
#         for i in range(len(fault_type_list)):
#             index_dict[i] = np.where(temp_y == i)[0]
#             k = max(k, index_dict[i].shape[0])
#         for i in range(len(fault_type_list)):
#             index_dict[i] = random.choices(population=index_dict[i].tolist(), k=k)

#         indices = []
#         for i in range(len(fault_type_list)):
#             indices.extend(index_dict[i])
#         random.shuffle(indices)
#         indices = np.array(indices)

#         x = x[indices]
#         return np.array(x).transpose().tolist()


# from sklearn.model_selection import train_test_split
# import numpy as np
# import pickle
# import random

# # --- 固定随机种子 ---
# random.seed(409)
# np.random.seed(409)

# class DatasetHandler:
#     @staticmethod
#     def split_and_save_dataset(modal_type_list: list,
#                                modal_data: dict,
#                                ent_edge_index: dict,
#                                valid_ratio: float, # 例如 0.1
#                                y: dict,
#                                multi_class_label_format: bool,
#                                num_of_fault_types: int,
#                                meta_data: dict,
#                                save_file_path: str):
#         result_dict = {
#             'data': dict(),
#             'meta_data': meta_data
#         }

#         test_size = valid_ratio
#         random_state = 409

#         # --- 1. 提取 fault 和 normal 的 train_valid/test 数据 ---
#         fault_train_valid_list, fault_test_list = [], []
#         normal_train_valid_list, normal_test_list = [], []

#         for modal_type in modal_type_list:
#             fault_train_valid_list.append(modal_data[modal_type]['train_valid'])
#             fault_test_list.append(modal_data[modal_type]['test'])
#         fault_train_valid_list.append(ent_edge_index['train_valid'])
#         fault_test_list.append(ent_edge_index['test'])
#         fault_train_valid_list.append(y['train_valid'])
#         fault_test_list.append(y['test'])

#         for modal_type in modal_type_list:
#             normal_train_valid_list.append(modal_data[modal_type]['normal_for_train_valid'])
#             normal_test_list.append(modal_data[modal_type]['normal_for_test'])
#         normal_train_valid_list.append(ent_edge_index['normal_for_train_valid'])
#         normal_test_list.append(ent_edge_index['normal_for_test'])
#         normal_train_valid_list.append(y['normal_for_train_valid'])
#         normal_test_list.append(y['normal_for_test'])

#         # 验证长度是否一致
#         assert len(fault_train_valid_list[0]) == len(normal_train_valid_list[0]), \
#             f"Mismatch in train_valid lengths: fault={len(fault_train_valid_list[0])}, normal={len(normal_train_valid_list[0])}"
#         assert len(fault_test_list[0]) == len(normal_test_list[0]), \
#             f"Mismatch in test lengths: fault={len(fault_test_list[0])}, normal={len(normal_test_list[0])}"

#         original_train_valid_size = len(fault_train_valid_list[0])

#         # --- 2. 按元素合并 fault 和 normal 数据 (创建新的复合样本) ---
#         combined_train_valid_list = []
#         combined_test_list = []

#         # 合并 train_valid
#         for i in range(len(fault_train_valid_list)):
#             combined_component = []
#             for j in range(original_train_valid_size):
#                 combined_component.append((fault_train_valid_list[i][j], normal_train_valid_list[i][j]))
#             combined_train_valid_list.append(combined_component)

#         # 合并 test
#         for i in range(len(fault_test_list)):
#             combined_component = []
#             for j in range(len(fault_test_list[i])):
#                 combined_component.append((fault_test_list[i][j], normal_test_list[i][j]))
#             combined_test_list.append(combined_component)

#         # --- 3. 对合并后的复合数据集进行划分 ---
#         combined_train_valid_split = train_test_split(*tuple(combined_train_valid_list),
#                                                       test_size=test_size,
#                                                       random_state=random_state,
#                                                       shuffle=True)
#         combined_data = dict()
#         combined_data['train'] = combined_train_valid_split[::2] # [train_part0, train_part1, ...]
#         combined_data['valid'] = combined_train_valid_split[1::2] # [valid_part0, valid_part1, ...]
#         combined_data['test'] = combined_test_list # Test 数据不变

#         # --- 4. 计算或确定目标 k 值 ---
#         # 假设你知道期望的重采样大小是 384，并且类别数是 3 (你需要根据实际情况调整)
#         # 那么 k = 384 / 3 = 128
#         # 或者，你从原始的 over_sampling 逻辑中推断出 k 应该是 128
#         # 为了更通用，我们可以基于原始的 fault_train_valid_list 来计算它
#         original_fault_labels = fault_train_valid_list[-1] # Extract original fault labels
#         temp_y_orig = []
#         fault_type_list_orig = []
#         for label in original_fault_labels:
#             fault_type = 'None'
#             if np.count_nonzero(label) > 1:
#                 fault_type = f'service:{int(np.max(label))}'
#             elif np.count_nonzero(label) == 1:
#                 fault_type = f'{int(np.max(label))}'
#             if fault_type not in fault_type_list_orig:
#                 fault_type_list_orig.append(fault_type)
#             temp_y_orig.append(fault_type_list_orig.index(fault_type))
#         temp_y_orig = np.array(temp_y_orig, dtype=object)

#         k_target = 0
#         for i in range(len(fault_type_list_orig)):
#             indices = np.where(temp_y_orig == i)[0]
#             k_target = max(k_target, indices.shape[0])
#         # k_target 现在是基于原始数据计算出的 k
#         print(f"Calculated target k based on original data: {k_target}")

#         # --- 5. 对合并后的训练集进行过采样 (使用目标 k) ---
#         combined_data['train'], _ = DatasetHandler.over_sampling_combined(combined_data['train'], k_target)

#         # --- 6. 分离 fault 和 normal 数据 ---
#         # 将复合样本 (fault_sample, normal_sample) 分离回 fault 和 normal 列表
#         fault_data = {'train': [], 'valid': [], 'test': []}
#         normal_data = {'train': [], 'valid': [], 'test': []}

#         for data_type in ['train', 'valid', 'test']:
#             for i in range(len(combined_data[data_type])):
#                 fault_component = []
#                 normal_component = []
#                 for combined_sample in combined_data[data_type][i]:
#                     fault_sample, normal_sample = combined_sample
#                     fault_component.append(fault_sample)
#                     normal_component.append(normal_sample)
#                 fault_data[data_type].append(fault_component)
#                 normal_data[data_type].append(normal_component)

#         # --- 7. 原有的数据校验逻辑 (针对 fault 数据) ---
#         for i, modal_name in enumerate(modal_type_list):
#             print(f"\n=== Modal {modal_name} (index {i}) ===")
#             shapes = []
#             for sample in fault_data['train'][i]:
#                 if hasattr(sample, 'shape'):
#                     shapes.append(sample.shape)
#                 else:
#                     shapes.append(('not array', type(sample)))
#             from collections import Counter
#             shape_counts = Counter(shapes)
#             print("Unique shapes in fault train:", shape_counts)

#             valid_shapes = [s.shape if hasattr(s, 'shape') else ('not array', type(s)) for s in fault_data['valid'][i]]
#             valid_shape_counts = Counter(valid_shapes)
#             print("Unique shapes in fault valid:", valid_shape_counts)

#         # --- 8. 原有的数据保存逻辑 ---
#         for data_type in ['train', 'valid', 'test']:
#             # 保存 fault 数据
#             for i in range(len(modal_type_list)):
#                 try:
#                     arr = np.stack(fault_data[data_type][i])
#                 except ValueError as e:
#                     print(f"Shape mismatch in fault {data_type}, modal {i}: {e}")
#                     arr = np.array(fault_data[data_type][i], dtype=object)
#                 result_dict['data'][f'x_{modal_type_list[i]}_{data_type}'] = arr
#             result_dict['data'][f'ent_edge_index_{data_type}'] = fault_data[data_type][len(modal_type_list)]

#             if multi_class_label_format:
#                 fault_data[data_type][len(modal_type_list) + 1] = DatasetHandler.label_to_multi_class_format(fault_data[data_type][len(modal_type_list) + 1], num_of_fault_types)

#             result_dict['data'][f'y_{data_type}'] = np.array(fault_data[data_type][len(modal_type_list) + 1])

#             # 保存 normal 数据
#             for i in range(len(modal_type_list)):
#                 try:
#                     normal_arr = np.stack(normal_data[data_type][i])
#                 except ValueError as e:
#                     print(f"Shape mismatch in normal {data_type}, modal {i}: {e}")
#                     normal_arr = np.array(normal_data[data_type][i], dtype=object)
#                 result_dict['data'][f'x_{modal_type_list[i]}_{data_type}_normal'] = normal_arr
#             result_dict['data'][f'ent_edge_index_{data_type}_normal'] = normal_data[data_type][len(modal_type_list)]
#             if multi_class_label_format and len(normal_data[data_type]) > len(modal_type_list) + 1:
#                 normal_data[data_type][len(modal_type_list) + 1] = DatasetHandler.label_to_multi_class_format(normal_data[data_type][len(modal_type_list) + 1], num_of_fault_types)
#             result_dict['data'][f'y_{data_type}_normal'] = np.array(normal_data[data_type][len(modal_type_list) + 1])

#         with open(save_file_path, 'wb') as f:
#             pickle.dump(result_dict, f, protocol=4)

#     @staticmethod
#     def label_to_multi_class_format(raw_y, num_of_fault_types=15):
#         y = []
#         raw_y = np.array(raw_y)
#         for i in range(raw_y.shape[0]):
#             y.append([])
#             for j in range(raw_y.shape[1]):
#                 y[-1].append(np.zeros(num_of_fault_types))
#                 if raw_y[i][j] != 0:
#                     y[-1][-1][int(raw_y[i][j] - 1)] = 1
#         return np.array(y)

#     @staticmethod
#     def over_sampling_combined(combined_train_list, target_k=None):
#         """
#         Modified over_sampling that works on combined (fault, normal) samples.
#         Uses the fault sample's label to determine the sampling strategy.
#         If target_k is provided, it uses that value instead of calculating k from the current data.
#         """
#         y_combined = combined_train_list[-1]
#         y_fault = [pair[0] for pair in y_combined]

#         temp_y = []
#         fault_type_list = []
#         for label in y_fault:
#             fault_type = 'None'
#             if np.count_nonzero(label) > 1:
#                 fault_type = f'service:{int(np.max(label))}'
#             elif np.count_nonzero(label) == 1:
#                 fault_type = f'{int(np.max(label))}'
#             if fault_type not in fault_type_list:
#                 fault_type_list.append(fault_type)
#             temp_y.append(fault_type_list.index(fault_type))

#         temp_y = np.array(temp_y, dtype=object)

#         # Determine k: use provided target_k or calculate from current data
#         if target_k is None:
#             k = 0
#             for i in range(len(fault_type_list)):
#                 indices = np.where(temp_y == i)[0]
#                 k = max(k, indices.shape[0])
#         else:
#             k = target_k
#             print(f"Using provided target k: {k}")

#         index_dict = dict()
#         for i in range(len(fault_type_list)):
#             indices = np.where(temp_y == i)[0]
#             index_dict[i] = indices.tolist()
#             # Use the determined k for resampling
#             index_dict[i] = random.choices(population=index_dict[i], k=k)

#         indices = []
#         for i in range(len(fault_type_list)):
#             indices.extend(index_dict[i])
#         random.shuffle(indices)
#         indices = np.array(indices)

#         # Apply the same indices to ALL components
#         combined_train_oversampled = []
#         for component_list in combined_train_list:
#             oversampled_component = [component_list[idx] for idx in indices]
#             combined_train_oversampled.append(oversampled_component)

#         return combined_train_oversampled, indices
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import random


random.seed(409)


class DatasetHandler:
    @staticmethod
    def split_and_save_dataset(modal_type_list: list,
                               modal_data: dict,
                               ent_edge_index: dict,
                               valid_ratio: float,
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

        train_valid_list, test_list = [], []
        for modal_type in modal_type_list:
            train_valid_list.append(modal_data[modal_type]['train_valid'])
            test_list.append(modal_data[modal_type]['test'])
        train_valid_list.append(ent_edge_index['train_valid'])
        test_list.append(ent_edge_index['test'])
        train_valid_list.append(y['train_valid'])
        test_list.append(y['test'])

        train_valid = train_test_split(*tuple(train_valid_list),
                                       test_size=test_size,
                                       random_state=random_state,
                                       shuffle=True)

        data = dict()
        data['train'], data['valid'], data['test'] = train_valid[::2], train_valid[1::2], test_list
        # print("####  size: ", len(data['train'][0]))
        data['train'] = DatasetHandler.over_sampling(data['train'])
        # print("####  size: ", len(data['train'][0]))
        
        ## =========== test whether there are abnormal data =================
        for i, modal_name in enumerate(modal_type_list):
            print(f"\n=== Modal {modal_name} (index {i}) ===")
            shapes = []
            for sample in data['train'][i]:
                if hasattr(sample, 'shape'):
                    shapes.append(sample.shape)
                else:
                    shapes.append(('not array', type(sample)))
            
            # 统计 unique shapes
            from collections import Counter
            shape_counts = Counter(shapes)
            print("Unique shapes in train:", shape_counts)

            # 对比 valid
            valid_shapes = [s.shape if hasattr(s, 'shape') else ('not array', type(s)) for s in data['valid'][i]]
            valid_shape_counts = Counter(valid_shapes)
            print("Unique shapes in valid:", valid_shape_counts)
            
        ## =================================
        
        ## for those with more than two shapes, please run the following code.
        ## 
        # def align_sequence_length_0(samples, target_len=17):
        #     aligned = []
        #     for s in samples:
        #         if s.shape[0] > target_len:
        #             # 截断（保留前 target_len 步）
        #             aligned.append(s[:target_len])
        #         elif s.shape[0] < target_len:
        #             # 填充（用0）
        #             pad_width = ((0, target_len - s.shape[0]), (0, 0))
        #             padded = np.pad(s, pad_width, mode='constant', constant_values=0)
        #             aligned.append(padded)
        #         else:
        #             aligned.append(s)
        #     return aligned
        
        # import numpy as np

        # def align_sequence_length_1(samples, target_len=17):
        #     aligned = []
        #     for s in samples:
        #         current_len = s.shape[0]
        #         if current_len > target_len:
        #             # 截断：保留前 target_len 步
        #             aligned.append(s[:target_len])
        #         elif current_len < target_len:
        #             # 填充：用最后一帧重复补齐
        #             last_frame = s[-1:]  # 保持维度 (1, F)
        #             repeat_times = target_len - current_len
        #             padded = np.concatenate([s, np.repeat(last_frame, repeat_times, axis=0)], axis=0)
        #             aligned.append(padded)
        #         else:
        #             # 长度正好，直接保留
        #             aligned.append(s)
        #     return aligned
        
        # for data_type in ['train', 'valid', 'test']:
        #     for i in range(len(modal_type_list)):
        #         data[data_type][i] = align_sequence_length_1(data[data_type][i], target_len=17)

        for data_type in ['train', 'valid', 'test']:
            for i in range(len(modal_type_list)):
                # print(f"i: {i}")
                # print(f'x_{modal_type_list[i]}')
                # print(len(data[data_type][i]))
                # print(len(data[data_type][i][0][0]))
                # data_tmp = np.array(data[data_type][i], dtype=object)
                # print(data_tmp.shape)
                # result_dict['data'][f'x_{modal_type_list[i]}_{data_type}'] = data_tmp
                # result_dict['data'][f'x_{modal_type_list[i]}_{data_type}'] = np.array(data[data_type][i], dtype=object) # old
                try:
                    arr = np.stack(data[data_type][i])  # 要求所有样本 shape 一致
                except ValueError as e:
                    print(f"Shape mismatch in {data_type}, modal {i}: {e}")
                    # fallback to object array or debug
                    arr = np.array(data[data_type][i], dtype=object)
                result_dict['data'][f'x_{modal_type_list[i]}_{data_type}'] = arr
            result_dict['data'][f'ent_edge_index_{data_type}'] = data[data_type][len(modal_type_list)]

            if multi_class_label_format:
                data[data_type][len(modal_type_list) + 1] = DatasetHandler.label_to_multi_class_format(data[data_type][len(modal_type_list) + 1], num_of_fault_types)

            result_dict['data'][f'y_{data_type}'] = np.array(data[data_type][len(modal_type_list) + 1])

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
    def over_sampling(train):
        x = np.array(train, dtype=object).transpose().tolist()
        y_train = train[-1]

        temp_y = []
        fault_type_list = []
        for label in y_train:
            fault_type = 'None'
            if np.count_nonzero(label) > 1:
                fault_type = f'service:{int(np.max(label))}'
            elif np.count_nonzero(label) == 1:
                fault_type = f'{int(np.max(label))}'
            if fault_type not in fault_type_list:
                fault_type_list.append(fault_type)
            temp_y.append(fault_type_list.index(fault_type))
        x, temp_y = np.array(x, dtype=object), np.array(temp_y, dtype=object)

        k = 0
        index_dict = dict()
        for i in range(len(fault_type_list)):
            index_dict[i] = np.where(temp_y == i)[0]
            k = max(k, index_dict[i].shape[0])
        for i in range(len(fault_type_list)):
            index_dict[i] = random.choices(population=index_dict[i].tolist(), k=k)

        indices = []
        for i in range(len(fault_type_list)):
            indices.extend(index_dict[i])
        random.shuffle(indices)
        indices = np.array(indices)

        x = x[indices]
        return np.array(x).transpose().tolist()

# IncluRCA/data_loader/mask_learning_data_loader.py (or wherever it's defined)
import pickle
from torch.utils.data import DataLoader

from IncluRCA.base.base_data_loader import BaseDataLoader
from IncluRCA.dataset.rca_dataset import RCADataset


class MaskLearningDataLoader(BaseDataLoader):
    def __init__(self, param_dict, dataset_type='test'): # Add dataset_type parameter
        super().__init__(param_dict)
        self.dataset_type = dataset_type # Store the dataset type

    def load_data(self, data_path):
        with open(f'{data_path}', 'rb') as f:
            temp = pickle.load(f)
        self.meta_data = temp['meta_data']
        data = dict()
        # Use the specified dataset_type instead of hardcoded 'test'
        for modal_type in self.meta_data['modal_types']:
            data[f'x_{modal_type}'] = temp['data'][f'x_{modal_type}_{self.dataset_type}'].transpose((0, 2, 1))
        data[f'ent_edge_index'] = temp['data'][f'ent_edge_index_{self.dataset_type}']
        data[f'y'] = temp['data'][f'y_{self.dataset_type}']
        self.data_loader = DataLoader(RCADataset(data), batch_size=1, shuffle=False)

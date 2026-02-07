import pickle
from torch.utils.data import DataLoader
import numpy as np

from IncluRCA.base.base_data_loader import BaseDataLoader
from IncluRCA.dataset.rca_dataset import RCADataset


class RCADataLoader(BaseDataLoader):
    def __init__(self, param_dict):
        super().__init__(param_dict)

    def load_data(self, data_path):
        with open(f'{data_path}', 'rb') as f:
            temp = pickle.load(f)
        self.meta_data = temp['meta_data']
        data = dict()

        # Identify available dataset types from the loaded data
        # Check for ALL modalities defined in meta_data for each potential split
        available_keys = set(temp['data'].keys())
        
        # Determine which splits (train, valid, test) have ALL modalities present
        dataset_types = []
        for dt in ['train', 'valid', 'test']:
            all_modalities_present = True
            for modal_type in self.meta_data['modal_types']:
                required_key = f'x_{modal_type}_{dt}'
                if required_key not in available_keys:
                    all_modalities_present = False
                    print(f"Warning: Key '{required_key}' not found for split '{dt}'. Skipping this split.")
                    break # Exit inner loop if any modality is missing for this split
            if all_modalities_present:
                dataset_types.append(dt)
        
        # Determine which normal splits (train_normal, valid_normal, test_normal) have ALL modalities present
        normal_dataset_types = []
        for dt in ['train', 'valid', 'test']: # Iterate over base names again
            all_modalities_present = True
            for modal_type in self.meta_data['modal_types']:
                required_key = f'x_{modal_type}_{dt}_normal' # Check for _normal suffix
                if required_key not in available_keys:
                    all_modalities_present = False
                    print(f"Warning: Key '{required_key}' not found for normal split '{dt}_normal'. Skipping this split.")
                    break # Exit inner loop if any modality is missing for this normal split
            if all_modalities_present:
                normal_dataset_types.append(f"{dt}_normal") # Store the full name with _normal

        print(f"Available standard datasets (all modalities present): {dataset_types}")
        print(f"Available normal datasets (all modalities present): {normal_dataset_types}")

        # Process standard (fault) datasets
        for dataset_type in dataset_types:
            data[dataset_type] = dict()
            for modal_type in self.meta_data['modal_types']:
                key = f'x_{modal_type}_{dataset_type}'
                data[dataset_type][f'x_{modal_type}'] = temp['data'][key].transpose((0, 2, 1))
            
            # Load ent_edge_index (list of shape (N, 2, 164))
            ent_edge_key = f'ent_edge_index_{dataset_type}'
            data[dataset_type][f'ent_edge_index'] = temp['data'][ent_edge_key] # Keep as list
            
            # Load y (shape (N, 56, 15))
            y_key = f'y_{dataset_type}'
            data[dataset_type][f'y'] = temp['data'][y_key]

            shuffle = dataset_type == 'train'
            # Create DataLoader for standard fault-only datasets (if needed for specific evaluation)
            self.data_loader[dataset_type] = DataLoader(
                RCADataset(data[dataset_type]),
                batch_size=self.param_dict['batch_size'],
                shuffle=shuffle
            )

        # Process normal datasets
        data_normal = {}
        for normal_type in normal_dataset_types:
            base_type = normal_type.replace('_normal', '') # e.g., 'train_normal' -> 'train'
            data_normal[base_type] = dict()
            for modal_type in self.meta_data['modal_types']:
                key = f'x_{modal_type}_{normal_type}'
                data_normal[base_type][f'x_{modal_type}'] = temp['data'][key].transpose((0, 2, 1))
            
            ent_edge_key = f'ent_edge_index_{normal_type}'
            data_normal[base_type][f'ent_edge_index'] = temp['data'][ent_edge_key] # Keep as list
            
            y_key = f'y_{normal_type}'
            data_normal[base_type][f'y'] = temp['data'][y_key]

            shuffle = base_type == 'train'
            # Create DataLoader for normal-only datasets (if needed for specific evaluation or pre-training)
            self.data_loader[normal_type] = DataLoader(
                RCADataset(data_normal[base_type]),
                batch_size=self.param_dict['batch_size'],
                shuffle=shuffle
            )

        # Combine fault and normal datasets for training/evaluation
        # Only combine splits that exist in both standard and normal forms
        for base_type in dataset_types:
            normal_type = f"{base_type}_normal"
            if normal_type in normal_dataset_types: # Check if corresponding normal split exists and is valid
                combined_data = dict()

                # Combine x modalities using np.concatenate along axis 0 (batch dimension)
                for modal_type in self.meta_data['modal_types']:
                    x_fault = data[base_type][f'x_{modal_type}'] # Shape after transpose: (N_fault, F, T)
                    x_normal = data_normal[base_type][f'x_{modal_type}'] # Shape after transpose: (N_normal, F, T)
                    combined_data[f'x_{modal_type}'] = np.concatenate([x_fault, x_normal], axis=0)

                # Combine ent_edge_index lists
                ent_edge_fault = data[base_type][f'ent_edge_index'] # List of length N_fault
                ent_edge_normal = data_normal[base_type][f'ent_edge_index'] # List of length N_normal
                combined_data[f'ent_edge_index'] = ent_edge_fault + ent_edge_normal # Concatenated list

                # Combine y labels using np.concatenate along axis 0 (batch dimension)
                y_fault = data[base_type][f'y'] # Shape: (N_fault, 56, 15)
                y_normal = data_normal[base_type][f'y'] # Shape: (N_normal, 56, 15)
                combined_data[f'y'] = np.concatenate([y_fault, y_normal], axis=0)

                shuffle = base_type == 'train'
                # Create DataLoader for the combined dataset (fault + normal)
                # This is the primary DataLoader for training if the model can handle mixed data
                self.data_loader[f'{base_type}_combined'] = DataLoader(
                    RCADataset(combined_data), # Pass the combined structure, RCADataset handles x, ent_edge_index, y
                    batch_size=self.param_dict['batch_size'],
                    shuffle=shuffle
                )
                print(f"Created combined DataLoader for {base_type} with {len(combined_data['y'])} samples.")

        # Print final data_loader keys to confirm
        print(f"Final data_loader keys: {list(self.data_loader.keys())}")
import torch
import numpy as np  # Import numpy for sorting
from IncluRCA.data_loader.mask_learning_data_loader import MaskLearningDataLoader
from IncluRCA.base.base_rca_trainer_normal import BaseRCATrainer  # Import the new trainer class
from IncluRCA.util.data_handler import copy_batch_data
from IncluRCA.util.data_handler import rearrange_y
from shared_util.evaluation_metrics import *
from IncluRCA.explain.explainer_normal import Explainer

class BaseLocalizer(BaseRCATrainer):  # Inherits from the new normal-aware trainer
    def __init__(self, param_dict):
        super().__init__(param_dict)  # Call parent __init__
        self.mask_learning_data_loader = MaskLearningDataLoader(param_dict)
        self.mask_learning_data_loader.load_data(f'{self.param_dict["dataset_path"]}')

    def predict(self):
        # Load the full model state (including normal/abnormal classifier)
        checkpoint = torch.load(self.param_dict["model_path"])
        self.representation_learning.load_state_dict(checkpoint['representation_learning'])
        self.feature_integration.load_state_dict(checkpoint['feature_integration'])
        self.feature_fusion.load_state_dict(checkpoint['feature_fusion'])
        self.fault_classifier.load_state_dict(checkpoint['fault_classifier'])
        self.normal_abnormal_classifier.load_state_dict(checkpoint['normal_abnormal_classifier'])

        # Critical: Set model to train mode (needed for gradient computation in explainer)
        self.representation_learning.train()
        self.feature_integration.train()
        self.feature_fusion.train()
        self.fault_classifier.train()
        self.normal_abnormal_classifier.train()

        explainer = Explainer(self, self.mask_learning_data_loader.meta_data, self.param_dict)
        result = dict()

        # Initialize result structure
        for ent_type in self.rca_data_loader.meta_data['ent_types']:
            if ent_type not in result.keys():
                result[ent_type] = {
                    'total': 0,
                    'd1': {
                        'AC@1_num': 0,
                        'AC@3_num': 0,
                        'AC@5_num': 0,
                    },
                    'd2': {
                        'AC@1_num': 0,
                        'AC@3_num': 0,
                        'AC@5_num': 0,
                    }
                }

        # Iterate through data loader
        for batch_id, batch_data in enumerate(self.mask_learning_data_loader.data_loader):
            explain_batch_data = copy_batch_data(batch_data, self.device)
            y = rearrange_y(self.rca_data_loader.meta_data, batch_data['y'], self.device)
            root_cause_list = []
            
            # Extract root cause list (FIXED: np.nonzero index handling)
            for ent_type in y.keys():
                # Get non-zero indices (handle 1D/2D arrays)
                y_ent = y[ent_type].cpu().detach().numpy()  # Convert to numpy array
                pos = np.nonzero(y_ent)  # Get non-zero indices
                
                # Handle different array dimensions
                if len(pos) == 1:
                    # 1D array: only row indices
                    for i in range(len(pos[0])):
                        row_idx = pos[0][i]
                        # For 1D array, use row index as fault type index (adjust based on your data structure)
                        fault_type_idx = row_idx
                        root_cause_list.append({
                            'd1': self.mask_learning_data_loader.meta_data['ent_names'][
                                (self.mask_learning_data_loader.meta_data['ent_type_index'][ent_type][0] + row_idx).item()
                            ],
                            'd2': {
                                'exact': self.mask_learning_data_loader.meta_data['fault_type_related_o11y_names'][
                                    (self.mask_learning_data_loader.meta_data['ent_fault_type_index'][ent_type][0] + fault_type_idx).item()
                                ] if len(self.mask_learning_data_loader.meta_data['fault_type_related_o11y_names']) > 0 else [],
                                'fuzzy': []  # Add fuzzy matching if needed
                            },
                            'level': ent_type,
                            'fault_type': self.mask_learning_data_loader.meta_data['fault_type_list'][
                                self.mask_learning_data_loader.meta_data['ent_fault_type_index'][ent_type][0] + fault_type_idx
                            ] if (self.mask_learning_data_loader.meta_data['ent_fault_type_index'][ent_type][0] + fault_type_idx) < len(self.mask_learning_data_loader.meta_data['fault_type_list']) else ""
                        })
                elif len(pos) == 2:
                    # 2D array: row + column indices (original logic)
                    for i in range(len(pos[0])):
                        row_idx = pos[0][i]
                        col_idx = pos[1][i]
                        root_cause_list.append({
                            'd1': self.mask_learning_data_loader.meta_data['ent_names'][
                                (self.mask_learning_data_loader.meta_data['ent_type_index'][ent_type][0] + row_idx).item()
                            ],
                            'd2': self.mask_learning_data_loader.meta_data['fault_type_related_o11y_names'][(self.mask_learning_data_loader.meta_data['ent_fault_type_index'][ent_type][0] + col_idx).item()],
                            # 'd2': {
                            #     'exact': self.mask_learning_data_loader.meta_data['fault_type_related_o11y_names'][
                            #         (self.mask_learning_data_loader.meta_data['ent_fault_type_index'][ent_type][0] + col_idx).item()
                            #     ] if len(self.mask_learning_data_loader.meta_data['fault_type_related_o11y_names']) > 0 else [],
                            #     'fuzzy': []  # Add fuzzy matching if needed
                            # },
                            'level': ent_type,
                            'fault_type': self.mask_learning_data_loader.meta_data['fault_type_list'][
                                self.mask_learning_data_loader.meta_data['ent_fault_type_index'][ent_type][0] + col_idx
                            ] if (self.mask_learning_data_loader.meta_data['ent_fault_type_index'][ent_type][0] + col_idx) < len(self.mask_learning_data_loader.meta_data['fault_type_list']) else ""
                        })
                else:
                    # Unexpected dimension: skip
                    self.logger.warning(f"Unexpected dimension for {ent_type} y array: {len(pos)}")
                    continue

            # Get exact root cause (prioritize service level)
            exact_root_cause = {}
            for root_cause in root_cause_list:
                exact_root_cause = root_cause
                if root_cause['level'] == 'service':
                    break

            if not exact_root_cause:  # Handle empty root cause case
                self.logger.warning(f"No root cause found for batch {batch_id}")
                continue
            result[exact_root_cause['level']]['total'] += 1

            # Forward pass (no torch.no_grad() - needed for explainer gradient)
            fault_out, _, _ = self.forward_model(batch_data)

            localization_result = {
                'd1': dict(),
                'd2': dict()
            }
            suspect_list = []

            # Generate suspect list (entities with predicted faults)
            for ent_type in self.rca_data_loader.meta_data['ent_types']:
                fault_prob = torch.sigmoid(fault_out[ent_type])
                temp_y_pred = (fault_prob > self.param_dict[f'{ent_type}_accuracy_th']).cpu().detach().numpy()
                ent_fault_prob = torch.max(fault_prob, dim=1).values.cpu().detach().numpy()
                
                for ent_index in range(len(temp_y_pred)):
                    if temp_y_pred[ent_index].any():  # Only add entities with predicted faults
                        suspect_list.append((ent_index, ent_type, ent_fault_prob[ent_index]))

            # Process suspect list
            if suspect_list:
                # Sort suspects by fault probability (descending)
                suspect_list = sorted(suspect_list, key=lambda item: item[2], reverse=True)
                trigger_ent_index, trigger_ent_type, trigger_fault_prob = suspect_list[0]
                trigger_ent_name = self.mask_learning_data_loader.meta_data['ent_names'][
                    (self.mask_learning_data_loader.meta_data['ent_type_index'][trigger_ent_type][0] + trigger_ent_index)
                ]
                
                # Add trigger entity to localization result
                if trigger_ent_name not in localization_result['d1']:
                    localization_result['d1'][trigger_ent_name] = 0
                localization_result['d1'][trigger_ent_name] += trigger_fault_prob

                # Train explainer for trigger entity
                ent_name_result, o11y_name_result = explainer.train_explainer(
                    explain_batch_data,
                    torch.sigmoid(fault_out[trigger_ent_type])[trigger_ent_index],
                    trigger_ent_type,
                    trigger_ent_index
                )
                
                # Update localization results with explainer output
                for ent_name_pair in ent_name_result:
                    if ent_name_pair[1] not in localization_result['d1']:
                        localization_result['d1'][ent_name_pair[1]] = 0
                    localization_result['d1'][ent_name_pair[1]] += ent_name_pair[0] * trigger_fault_prob
                
                for o11y_name_pair in o11y_name_result:
                    if o11y_name_pair[1] not in localization_result['d2']:
                        localization_result['d2'][o11y_name_pair[1]] = 0
                    localization_result['d2'][o11y_name_pair[1]] += o11y_name_pair[0] * trigger_fault_prob

                # Clean GPU cache (optional but recommended)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            else:
                # Fallback: find entity with highest fault probability (even below threshold)
                max_prob = -1
                max_ent_idx, max_ent_type = -1, None
                
                for ent_type in self.rca_data_loader.meta_data['ent_types']:
                    fault_prob = torch.sigmoid(fault_out[ent_type])
                    ent_fault_prob = torch.max(fault_prob, dim=1).values.cpu().detach().numpy()
                    max_idx_in_batch = np.argmax(ent_fault_prob)
                    max_prob_in_batch = ent_fault_prob[max_idx_in_batch]
                    
                    if max_prob_in_batch > max_prob:
                        max_prob = max_prob_in_batch
                        max_ent_idx = max_idx_in_batch
                        max_ent_type = ent_type

                if max_ent_idx != -1 and max_ent_type:
                    # Train explainer for max prob entity
                    ent_name_result, o11y_name_result = explainer.train_explainer(
                        explain_batch_data,
                        torch.sigmoid(fault_out[max_ent_type])[max_ent_idx],
                        max_ent_type,
                        max_ent_idx
                    )
                    
                    trigger_ent_name = self.mask_learning_data_loader.meta_data['ent_names'][
                        (self.mask_learning_data_loader.meta_data['ent_type_index'][max_ent_type][0] + max_ent_idx)
                    ]
                    
                    if trigger_ent_name not in localization_result['d1']:
                        localization_result['d1'][trigger_ent_name] = 0
                    localization_result['d1'][trigger_ent_name] += max_prob

                    # Update localization results
                    for ent_name_pair in ent_name_result:
                        if ent_name_pair[1] not in localization_result['d1']:
                            localization_result['d1'][ent_name_pair[1]] = 0
                        localization_result['d1'][ent_name_pair[1]] += ent_name_pair[0] * max_prob
                    
                    for o11y_name_pair in o11y_name_result:
                        if o11y_name_pair[1] not in localization_result['d2']:
                            localization_result['d2'][o11y_name_pair[1]] = 0
                        localization_result['d2'][o11y_name_pair[1]] += o11y_name_pair[0] * max_prob

                    # Clean GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Continue explaining other suspects to fill top-5 lists
            suspect_index = 1
            while len(localization_result['d1']) < 5 or len(localization_result['d2']) < 5:
                if suspect_index >= len(suspect_list):
                    break
                
                ent_idx, ent_typ, ent_prob = suspect_list[suspect_index]
                ent_name_result, o11y_name_result = explainer.train_explainer(
                    explain_batch_data,
                    torch.sigmoid(fault_out[ent_typ])[ent_idx],
                    ent_typ,
                    ent_idx
                )
                
                # Update localization results
                for ent_name_pair in ent_name_result:
                    if ent_name_pair[1] not in localization_result['d1']:
                        localization_result['d1'][ent_name_pair[1]] = 0
                    localization_result['d1'][ent_name_pair[1]] += ent_name_pair[0] * ent_prob
                
                for o11y_name_pair in o11y_name_result:
                    if o11y_name_pair[1] not in localization_result['d2']:
                        localization_result['d2'][o11y_name_pair[1]] = 0
                    localization_result['d2'][o11y_name_pair[1]] += o11y_name_pair[0] * ent_prob
                
                suspect_index += 1

                # Clean GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Sort localization results by score (descending)
            localization_result['d1'] = sorted(localization_result['d1'].items(), key=lambda item: item[1], reverse=True)
            localization_result['d2'] = sorted(localization_result['d2'].items(), key=lambda item: item[1], reverse=True)

            # Log results for current sample
            self.logger.info('----------')
            self.logger.info(
                f'sample {batch_id}/{len(self.mask_learning_data_loader.data_loader)} | '
                f'd1: {exact_root_cause["d1"]}; level: {exact_root_cause["level"]}; fault_type: {exact_root_cause["fault_type"]}'
            )
            self.logger.info(
                f'sample {batch_id}/{len(self.mask_learning_data_loader.data_loader)} | '
                f'predict d1: {localization_result["d1"][0:min(5, len(localization_result["d1"]))]}'
            )
            self.logger.info(
                f'sample {batch_id}/{len(self.mask_learning_data_loader.data_loader)} | '
                f'predict d2: {localization_result["d2"][0:min(5, len(localization_result["d2"]))]}'
            )
            self.logger.info('----------')

            # Calculate hit metrics (AC@1, AC@3, AC@5)
            d1_hit, d2_hit = dict(), dict()
            k_list = [1, 3, 5]
            for k in k_list:
                d1_hit[k], d2_hit[k] = False, False

            # Check D1 hits
            for i in range(len(localization_result['d1'])):
                if exact_root_cause['d1'] in localization_result['d1'][i][0]:
                    for k in k_list:
                        if i < k:
                            d1_hit[k] = True

            # Check D2 hits (exact + fuzzy)
            for i in range(len(localization_result['d2'])):
                hit = False
                # Exact match
                for exact_o11y_name in exact_root_cause['d2']['exact']:
                    if (exact_root_cause['d1'] in localization_result['d2'][i][0] and 
                        exact_o11y_name in localization_result['d2'][i][0]):
                        hit = True
                        # break
                # Fuzzy match (if exact failed)
                if not hit:
                    for fuzzy_o11y_name in exact_root_cause['d2']['fuzzy']:
                        if fuzzy_o11y_name in localization_result['d2'][i][0]:
                            hit = True
                            # break
                # Update hit flags
                if hit:
                    for k in k_list:
                        if i < k:
                            d2_hit[k] = True

            # Update result metrics
            for k in k_list:
                if d1_hit[k]:
                    result[exact_root_cause['level']]['d1'][f'AC@{k}_num'] += 1
                if d2_hit[k]:
                    result[exact_root_cause['level']]['d2'][f'AC@{k}_num'] += 1

        # Log final evaluation results
        self.logger.info('----------')
        self.logger.info(f'Final Evaluation Results')
        for ent_type in result.keys():
            if ent_type == "tidb":
                continue  # Skip tidb if needed
            total = result[ent_type]['total']
            if total == 0:
                self.logger.info(f'{ent_type.ljust(8)} | No samples to evaluate')
                continue
            # D1 metrics
            self.logger.info(
                f'{ent_type.ljust(8)} d1 | AC@1: {result[ent_type]["d1"]["AC@1_num"] / total:.6f}; '
                f'AC@3: {result[ent_type]["d1"]["AC@3_num"] / total:.6f}; '
                f'AC@5: {result[ent_type]["d1"]["AC@5_num"] / total:.6f}'
            )
            # D2 metrics
            self.logger.info(
                f'{ent_type.ljust(8)} d2 | AC@1: {result[ent_type]["d2"]["AC@1_num"] / total:.6f}; '
                f'AC@3: {result[ent_type]["d2"]["AC@3_num"] / total:.6f}; '
                f'AC@5: {result[ent_type]["d2"]["AC@5_num"] / total:.6f}'
            )
        self.logger.info('----------')
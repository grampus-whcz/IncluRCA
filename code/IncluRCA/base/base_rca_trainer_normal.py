# IncluRCA/base/base_rca_trainer_normal.py
from abc import ABC
from shared_util.logger import Logger
import torch
import torch.nn.functional as F # Added for binary cross entropy with logits
# from IncluRCA.data_loader.rca_data_loader import RCADataLoader
from IncluRCA.data_loader.rca_data_loader_normal import RCADataLoader
from IncluRCA.model.o11y.representation_learning import RepresentationLearning
# from IncluRCA.model.re.feature_integration import FeatureIntegration # Assuming standard Transformer version is used, or modify as needed
from IncluRCA.model.re.feature_integration_SEAttention import FeatureIntegration
from IncluRCA.model.re.feature_fusion import FeatureFusion
from IncluRCA.model.re.fault_classifier import FaultClassifier
from IncluRCA.util.data_handler import rearrange_y
from shared_util.evaluation_metrics import *
from torchinfo import summary
import numpy as np

class BaseRCATrainer(ABC):
    def __init__(self, param_dict):
        self.param_dict = param_dict
        assert param_dict['orl_te_in_channels'] == param_dict['efi_in_dim'] and param_dict['efi_out_dim'] == param_dict['eff_in_dim']
        self.device = torch.device("cuda" if torch.cuda.is_available() and param_dict['gpu'] else "cpu")
        self.logger = Logger(logging_level='DEBUG').logger
        self.rca_data_loader = RCADataLoader(param_dict)
        self.rca_data_loader.load_data(f'{self.param_dict["dataset_path"]}')

        o11y_representation_learning = RepresentationLearning(param_dict, self.rca_data_loader.meta_data)
        re_feature_integration = FeatureIntegration(param_dict, self.rca_data_loader.meta_data)
        re_feature_fusion = FeatureFusion(param_dict, self.rca_data_loader.meta_data)
        # Add a binary classifier head for normal/abnormal classification
        # This can be a simple linear layer followed by sigmoid
        # Ensure it's moved to the device immediately after creation
        self.normal_abnormal_classifier = torch.nn.Linear(param_dict['eff_GAT_out_channels'], 1).to(self.device)

        re_fault_classifier = FaultClassifier(param_dict, self.rca_data_loader.meta_data)

        # Note: The sequential model now needs custom forward logic to handle the new head
        # We will not use torch.nn.Sequential for the full model anymore
        self.representation_learning = o11y_representation_learning.to(self.device) # Ensure these are also on device
        self.feature_integration = re_feature_integration.to(self.device)
        self.feature_fusion = re_feature_fusion.to(self.device)
        self.fault_classifier = re_fault_classifier.to(self.device)
        # self.model = torch.nn.Sequential(o11y_representation_learning, re_feature_integration, re_feature_fusion, re_fault_classifier).to(self.device)
        self.model_rank = [] # summary(self.model)

    def forward_model(self, batch_data):
        """Custom forward pass incorporating the normal/abnormal classifier."""
        # Step 1: Representation Learning
        batch_data = self.representation_learning(batch_data)
        # Step 2: Feature Integration
        batch_data = self.feature_integration(batch_data)
        # Step 3: Feature Fusion (GAT)
        gat_output = self.feature_fusion(batch_data) # This returns the GAT output (B, num_entities, eff_GAT_out_channels)
        # Step 4: Fault Classification
        fault_out = self.fault_classifier(gat_output) # Dict of {ent_type: logits}
        # Step 5: Normal/Abnormal Classification
        # gat_output shape: (B, num_entities, eff_GAT_out_channels)
        # We can pool over entities (e.g., mean) or use a specific node's representation if applicable.
        # For simplicity, let's use mean pooling over entities for the entire batch.
        pooled_gat_output = gat_output.mean(dim=1) # Shape: (B, eff_GAT_out_channels)
        normal_abnormal_logits = self.normal_abnormal_classifier(pooled_gat_output).squeeze(-1) # Shape: (B,)
        return fault_out, normal_abnormal_logits, gat_output # Return gat_output if needed later (e.g., for loss weighting)

    def train(self):
        # --- Pre-training Phase (Optional) ---
        if self.param_dict.get('use_pretraining', False):
            self.logger.info("Starting Pre-training Phase...")
            pretrain_optimizer = torch.optim.Adam(
                list(self.representation_learning.parameters()) +
                list(self.feature_integration.parameters()),
                lr=self.param_dict['pretrain_lr'], # Add 'pretrain_lr' to param_dict
                weight_decay=self.param_dict['weight_decay']
            )
            # Example self-supervised task: Reconstruction Loss (requires modification in RepresentationLearning)
            # Or simply train the normal/abnormal classifier on normal data
            for epoch in range(self.param_dict['pretrain_epochs']): # Add 'pretrain_epochs' to param_dict
                self.representation_learning.train()
                self.feature_integration.train()
                self.feature_fusion.eval() # Freeze GAT and downstream
                self.fault_classifier.eval()
                self.normal_abnormal_classifier.train()
                pretrain_loss = 0
                for batch_id, batch_data in enumerate(self.rca_data_loader.data_loader['train_normal']): # Assumes data_loader has 'train_normal'
                    pretrain_optimizer.zero_grad()
                    _, na_logits, _ = self.forward_model(batch_data)
                    # Label: 0 for normal
                    # Infer batch size from an existing tensor in batch_data, e.g., 'y'
                    # Labels should be 0 for all samples in train_normal
                    na_labels = torch.zeros(batch_data['y'].shape[0], dtype=torch.float).to(self.device) # Shape: (B,)
                    # Use BCEWithLogitsLoss for numerical stability
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(na_logits, na_labels)
                    pretrain_loss += batch_data['y'].shape[0] * loss.item()
                    loss.backward()
                    pretrain_optimizer.step()
                pretrain_loss /= len(self.rca_data_loader.data_loader['train_normal'].dataset)
                self.logger.info(f'[Pre-train Epoch {epoch}/{self.param_dict["pretrain_epochs"]}] | Pre-train Loss: {pretrain_loss:.5f}')
            self.logger.info("Pre-training Phase Completed.")

        # --- Fine-tuning Phase ---
        self.logger.info("Starting Fine-tuning Phase...")
        # Optimizer for the full model or parts of it
        full_model_params = (
            list(self.representation_learning.parameters()) +
            list(self.feature_integration.parameters()) +
            list(self.feature_fusion.parameters()) +
            list(self.fault_classifier.parameters()) +
            list(self.normal_abnormal_classifier.parameters())
        )
        optimizer = torch.optim.Adam(full_model_params, lr=self.param_dict['lr'], weight_decay=self.param_dict['weight_decay'])

        # Loss functions
        criterion_dict = dict()
        for ent_type in self.rca_data_loader.meta_data['ent_types']:
            pos_weight = torch.FloatTensor(self.rca_data_loader.meta_data['ent_fault_type_weight'][ent_type]).to(self.device)
            criterion_dict[ent_type] = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none') # reduction='none' for masking later

        # Binary loss for normal/abnormal classification
        criterion_binary = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.param_dict.get('na_pos_weight', 1.0)).to(self.device))

        for epoch in range(self.param_dict['epochs']):
            self.representation_learning.train()
            self.feature_integration.train()
            self.feature_fusion.train()
            self.fault_classifier.train()
            self.normal_abnormal_classifier.train()

            train_loss = 0
            train_fault_loss = 0
            train_na_loss = 0

            # Iterate over both fault and normal data loaders
            # Option 1: Interleave batches from fault and normal sets
            # Option 2: Combine datasets beforehand (requires DataLoader modification)
            # We'll assume combined training data for simplicity here, where normal data has all fault labels as 0.
            # The data_loader['train'] should now contain both fault and normal samples.
            for batch_id, batch_data in enumerate(self.rca_data_loader.data_loader['train_combined']): # Assumes 'train_combined' exists
                optimizer.zero_grad()

                fault_out, na_logits, _ = self.forward_model(batch_data)

                # --- Prepare Labels for Loss Calculation ---
                # Raw labels from the combined dataset
                raw_y = batch_data['y'] # Shape: (B, num_entities, num_fault_types)

                # --- Calculate Fault Classification Loss ---
                # Use rearrange_y to get labels in the format expected by the criterion_dict
                # This function likely handles -1 masking for normal samples within faulty entities
                y_fault = rearrange_y(self.rca_data_loader.meta_data, raw_y, self.device) # Dict of {ent_type: (B, num_fault_types_for_ent_type)}
                fault_loss = 0
                for ent_type in self.rca_data_loader.meta_data['ent_types']:
                    # y_fault[ent_type] shape: (B, num_fault_types_for_ent_type)
                    # Check if there are any non-masked (non -1) labels in this batch for this entity type
                    # mask = ~(y_fault[ent_type] == -1).all(dim=1) # This masks samples where ALL fault types are -1 for this ent_type
                    # A more robust check: mask samples where all fault labels are effectively 0 (or masked -1)
                    # We assume rearrange_y correctly handles -1 masking internally in the loss calculation if reduction='none'
                    # However, we still need to sum losses only for valid entries per sample and type.
                    # The BCEWithLogitsLoss(reduction='none') will produce (B, num_fault_types_for_ent_type)
                    # We can mask entries where the target was -1 before summing across fault types.
                    mask = (y_fault[ent_type] != -1) # Boolean mask: True where label is not -1
                    if mask.any(): # Only calculate loss if there are valid (non -1) labels somewhere in the batch for this ent_type
                        loss_per_sample_per_type = criterion_dict[ent_type](
                            fault_out[ent_type], # (B, num_fault_types_for_ent_type)
                            y_fault[ent_type]    # (B, num_fault_types_for_ent_type)
                        ) # Shape: (B, num_fault_types_for_ent_type)
                        # Apply mask: set losses for -1 targets to 0
                        masked_loss_per_sample_per_type = loss_per_sample_per_type * mask.float()
                        # Sum loss across fault types for each sample
                        loss_per_sample = masked_loss_per_sample_per_type.sum(dim=1) # Shape: (B,)
                        # Mean loss across samples in the batch for this ent_type
                        fault_loss += loss_per_sample.mean() # Mean over samples in the batch for this ent_type


                # --- Calculate Normal/Abnormal Classification Loss ---
                # Infer normal/abnormal labels from raw_y
                # A sample is 'normal' if ALL its fault labels across ALL entities are 0 (or -1, depending on how normal data is labeled in 'y').
                # Assuming normal data in 'y' has all 0s for fault types.
                # Check if any fault label is 1 (indicating fault). If yes, label is 1. If all are 0/-1, label is 0.
                # raw_y shape: (B, num_entities, num_fault_types)
                # Check if any value along entity and fault_type dimensions is 1 (or > 0)
                # PyTorch version compatibility: use multiple .any() calls instead of dim=(1, 2)
                # any_fault = (raw_y > 0).any(dim=(1, 2)) # This line caused the error
                any_fault = (raw_y > 0).any(dim=2).any(dim=1) # Shape: (B,) - True if any fault exists in the sample

                # Convert boolean to float: True -> 1.0 (abnormal), False -> 0.0 (normal)
                na_labels = any_fault.float() # Shape: (B,) - 1.0 if fault, 0.0 if normal
                na_labels = na_labels.to(self.device) # Ensure labels are on the same device as logits

                na_loss = criterion_binary(na_logits, na_labels)

                # --- Combine Losses ---
                total_loss = self.param_dict.get('fault_loss_weight', 1.0) * fault_loss + \
                             self.param_dict.get('na_loss_weight', 1.0) * na_loss

                train_loss += batch_data['y'].shape[0] * total_loss.item()
                train_fault_loss += batch_data['y'].shape[0] * fault_loss.item() if fault_loss != 0 else 0
                train_na_loss += batch_data['y'].shape[0] * na_loss.item()

                total_loss.backward()
                optimizer.step()

            train_loss /= len(self.rca_data_loader.data_loader['train_combined'].dataset)
            train_fault_loss /= len(self.rca_data_loader.data_loader['train_combined'].dataset)
            train_na_loss /= len(self.rca_data_loader.data_loader['train_combined'].dataset)

            self.logger.info(f'[{epoch}/{self.param_dict["epochs"]}] | Total Loss: {train_loss:.5f} | Fault Loss: {train_fault_loss:.5f} | NA Loss: {train_na_loss:.5f}')

            # Validation (evaluate only on fault data)
            if epoch % 10 == 0:
                self.representation_learning.eval()
                self.feature_integration.eval()
                self.feature_fusion.eval()
                self.fault_classifier.eval()
                self.normal_abnormal_classifier.eval()

                y_pred = dict()
                y_true = dict()
                with torch.no_grad():
                    for batch_id, batch_data in enumerate(self.rca_data_loader.data_loader['valid']): # Use 'valid' which contains only fault data
                        y = rearrange_y(self.rca_data_loader.meta_data, batch_data['y'], self.device)
                        fault_out, _, _ = self.forward_model(batch_data)
                        for ent_type in self.rca_data_loader.meta_data['ent_types']:
                            if ent_type not in y_pred.keys():
                                y_pred[ent_type] = []
                                y_true[ent_type] = []
                            y_pred[ent_type].extend((torch.sigmoid(fault_out[ent_type][y[ent_type] != -1].reshape(-1, fault_out[ent_type].shape[1])) > self.param_dict[f'{ent_type}_accuracy_th']).cpu().detach().numpy())
                            y_true[ent_type].extend(y[ent_type][y[ent_type] != -1].reshape(-1, y[ent_type].shape[1]).cpu().detach().numpy())
                    self.output_evaluation_rca_d3_result(y_pred, y_true, 'valid')

        torch.save({
            'representation_learning': self.representation_learning.state_dict(),
            'feature_integration': self.feature_integration.state_dict(),
            'feature_fusion': self.feature_fusion.state_dict(),
            'fault_classifier': self.fault_classifier.state_dict(),
            'normal_abnormal_classifier': self.normal_abnormal_classifier.state_dict(),
            'optimizer': optimizer.state_dict()
        }, self.param_dict["model_path"])
        self.logger.info("Fine-tuning Phase Completed.")

    def evaluate_rca_d3(self):
        # Load the full saved state
        checkpoint = torch.load(self.param_dict["model_path"])
        self.representation_learning.load_state_dict(checkpoint['representation_learning'])
        self.feature_integration.load_state_dict(checkpoint['feature_integration'])
        self.feature_fusion.load_state_dict(checkpoint['feature_fusion'])
        self.fault_classifier.load_state_dict(checkpoint['fault_classifier'])
        self.normal_abnormal_classifier.load_state_dict(checkpoint['normal_abnormal_classifier'])

        self.representation_learning.eval()
        self.feature_integration.eval()
        self.feature_fusion.eval()
        self.fault_classifier.eval()
        self.normal_abnormal_classifier.eval()

        y_pred, y_true = dict(), dict()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(self.rca_data_loader.data_loader['test']): # Use 'test' which contains only fault data
                y = rearrange_y(self.rca_data_loader.meta_data, batch_data['y'], self.device)
                fault_out, _, _ = self.forward_model(batch_data)
                for ent_type in self.rca_data_loader.meta_data['ent_types']:
                    fault_prob = torch.sigmoid(fault_out[ent_type])
                    temp_y_pred = (fault_prob > self.param_dict[f'{ent_type}_accuracy_th']).cpu().detach().numpy()
                    temp_y_true = y[ent_type].cpu().detach().numpy()
                    if ent_type not in y_pred.keys():
                        y_pred[ent_type] = []
                        y_true[ent_type] = []
                    y_pred[ent_type].extend(temp_y_pred)
                    y_true[ent_type].extend(temp_y_true)
        self.output_evaluation_rca_d3_result(y_pred, y_true, 'test')

    def output_evaluation_rca_d3_result(self, y_pred, y_true, dataset_type):
        self.logger.info('----------')
        self.logger.info(f'evaluation dataset type: {dataset_type}')
        for ent_type in self.rca_data_loader.meta_data['ent_types']:
            ent_y_pred = np.array(y_pred[ent_type])
            ent_y_true = np.array(y_true[ent_type])
            # Handle case where there might be no true positives/negatives for an ent_type in the batch
            if ent_y_true.size == 0 or ent_y_pred.size == 0:
                self.logger.info(f'{ent_type.ljust(8)} precision/recall/f1 | micro: 0.000000; macro: 0.000000; score: N/A')
                continue
            try:
                fc_result = fault_type_classification(ent_y_pred, ent_y_true)
                convert = { 'p': 'precision', 'r': 'recall', 'f1': 'f1' }
                for em in ['p', 'r', 'f1']:
                    self.logger.info(f'{ent_type.ljust(8) + convert[em].ljust(9)} | micro: {fc_result["micro_" + convert[em] + "_score"]:.6f}; macro: {fc_result["macro_" + convert[em] + "_score"]:.6f}; score: {fc_result[convert[em] + "_score"]}')
            except ValueError as e:
                self.logger.error(f"Error in fault_type_classification for {ent_type}: {e}")
                self.logger.info(f'{ent_type.ljust(8)} precision/recall/f1 | Error during calculation')
        self.logger.info('----------')
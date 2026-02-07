import math
import torch
import torch.nn as nn
from torch.nn import Parameter, ParameterDict
from torch_geometric.nn import MessagePassing
from IncluRCA.util.data_handler import copy_batch_data
from torch_geometric.utils import sort_edge_index
import numpy as np
from shared_util.logger import Logger

class Explainer(nn.Module):
    def __init__(self, model, meta_data, param_dict):
        super().__init__()
        # Loss coefficients
        self.coeffs = {
            'ent_edge_size': 0.005,
            'ent_edge_reduction': 'sum',
            'o11y_size': 1.0,
            'o11y_reduction': 'mean',
            'ent_edge_entropy': 1.0,
            'o11y_entropy': 0.1,
            'EPS': 1e-15,
        }
        # Device configuration
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and param_dict['explainer_gpu'] else "cpu"
        )
        self.model = model
        self.meta_data = meta_data
        self.param_dict = param_dict
        # Mask variables
        self.o11y_mask = self.hard_o11y_mask = None
        self.ent_edge_mask = self.hard_ent_edge_mask = None
        # Loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()
        # Logger
        self.logger = Logger(logging_level='DEBUG').logger

    def init_o11y_mask(self):
        """Initialize observability feature mask"""
        mask = ParameterDict()
        for modal_type in self.meta_data['modal_types']:
            # Create parameter tensor for each modality
            mask[modal_type] = Parameter(
                torch.FloatTensor(self.meta_data['o11y_length'][modal_type])
            ).to(self.device)
            # Initialize with normal distribution
            std = 0.1
            with torch.no_grad():
                mask[modal_type].normal_(1.0, std)
        self.o11y_mask = mask

    def init_ent_edge_mask(self, test_sample_data):
        """Initialize entity edge mask"""
        num_edges = test_sample_data['ent_edge_index'].shape[2]
        num_entities = len(self.meta_data['ent_names'])
        # Calculate initialization std
        std = nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (num_entities + num_entities)
        )
        # Create edge mask parameter
        mask = Parameter(torch.randn(num_edges, device=self.device) * std)
        self.ent_edge_mask = mask

    def set_masks(self, test_sample_data):
        """Set masks to GAT network for edge explanation"""
        edge_index = torch.squeeze(test_sample_data['ent_edge_index'])
        # Access GAT network from feature fusion module
        gat_net = self.model.feature_fusion.GAT_net

        # Apply mask to MessagePassing layers (GATConv/GATv2Conv)
        for module in gat_net.modules():
            if isinstance(module, MessagePassing):
                loop_mask = torch.full_like(edge_index[0], True, dtype=bool)
                module.explain = True
                module._edge_mask = self.ent_edge_mask
                module._loop_mask = loop_mask
                module._apply_sigmoid = True

    def clean_explainer(self):
        """Clean up masks and GAT module settings"""
        # Reset GAT modules
        gat_net = self.model.feature_fusion.GAT_net
        for module in gat_net.modules():
            if isinstance(module, MessagePassing):
                module.explain = False
                module._edge_mask = None
                module._loop_mask = None
                module._apply_sigmoid = True
        # Reset masks
        self.o11y_mask = self.hard_o11y_mask = None
        self.ent_edge_mask = self.hard_ent_edge_mask = None

    def train_explainer(self, test_sample_data, original_y_pred, entity_type, entity_index):
        """Train explainer to find critical features/edges"""
        # Initialize masks
        self.init_o11y_mask()
        self.init_ent_edge_mask(test_sample_data)
        self.set_masks(test_sample_data)

        # Collect trainable parameters (masks)
        parameters = []
        for key in self.o11y_mask.keys():
            parameters.append(self.o11y_mask[key])
        parameters.append(self.ent_edge_mask)

        # Create optimizer for masks
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.param_dict['explainer_lr'],
            weight_decay=self.param_dict['explainer_weight_decay']
        )

        # Sort edge index for consistency
        test_sample_data['ent_edge_index'][0] = sort_edge_index(test_sample_data['ent_edge_index'][0])

        # Ensure original_y_pred is detached (avoid graph sharing)
        if not original_y_pred.requires_grad:
            with torch.enable_grad():
                temp_fault_out, _, _ = self.model.forward_model(test_sample_data)
                original_y_pred = temp_fault_out[entity_type][entity_index].detach().clone()
        else:
            original_y_pred = original_y_pred.detach().clone()

        # Get total training epochs
        total_epochs = self.param_dict['explainer_epochs']

        # Training loop
        for epoch in range(total_epochs):
            # Copy data (fresh copy each epoch to avoid graph reuse)
            copy_sample_data = copy_batch_data(test_sample_data, self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            # Zero model gradients (clean state)
            self.model.representation_learning.zero_grad()
            self.model.feature_integration.zero_grad()
            self.model.feature_fusion.zero_grad()
            self.model.fault_classifier.zero_grad()
            self.model.normal_abnormal_classifier.zero_grad()

            # Ensure input tensors have gradient enabled
            for key in copy_sample_data.keys():
                if (isinstance(copy_sample_data[key], torch.Tensor) and 
                    copy_sample_data[key].dtype in [torch.float32, torch.float64]):
                    copy_sample_data[key] = copy_sample_data[key].requires_grad_(True)

            # Apply mask to observability features
            for modal_type in self.meta_data['modal_types']:
                x = copy_sample_data[f'x_{modal_type}']
                # Generate noise for mask perturbation
                std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                z = torch.normal(mean=mean_tensor, std=std_tensor)
                # Apply mask
                copy_sample_data[f'x_{modal_type}'] = (
                    x + torch.mul(z.transpose(1, 2), (1 - self.o11y_mask[modal_type])).transpose(1, 2)
                )

            # Forward pass (fresh graph each epoch)
            fault_out, _, _ = self.model.forward_model(copy_sample_data)
            y_pred = fault_out[entity_type][entity_index]

            # Calculate loss
            loss = self.explainer_loss(y_pred, original_y_pred)

            # Log loss every 10 epochs
            if epoch % 10 == 0:
                self.logger.info(
                    f'[Explainer Epoch {epoch}/{total_epochs}] | train_loss: {loss.item():.5f}'
                )

            # Backward pass (retain graph except last epoch)
            retain_graph = (epoch < total_epochs - 1)
            loss.backward(retain_graph=retain_graph)
            
            # Update mask parameters
            optimizer.step()

        # Post-processing: extract hard masks from gradients
        if self.o11y_mask is not None:
            self.hard_o11y_mask = dict()
            for modal_type in self.meta_data['modal_types']:
                self.hard_o11y_mask[modal_type] = (self.o11y_mask[modal_type].grad != 0.0)

        if self.ent_edge_mask is not None:
            self.hard_ent_edge_mask = (self.ent_edge_mask.grad != 0.0)

        # Process edge mask results
        if self.hard_ent_edge_mask is not None and self.hard_ent_edge_mask.any():
            related_ent_edge_sorted_mask, related_ent_edge_indices = torch.sort(
                self.ent_edge_mask[self.hard_ent_edge_mask].sigmoid(), descending=True
            )
            final_edge_index = test_sample_data['ent_edge_index'][
                self.hard_ent_edge_mask.repeat(2, 1).unsqueeze(0)
            ].reshape(2, -1).t()[related_ent_edge_indices]

            # Generate entity name results
            ent_name_result, index_set = [], set()
            for i in range(final_edge_index.shape[0]):
                for j in [0, 1]:
                    idx_val = final_edge_index[i][j].cpu().detach().item()
                    if idx_val not in index_set:
                        ent_name = self.meta_data['ent_names'][idx_val]
                        ent_name_result.append(
                            (related_ent_edge_sorted_mask[i].cpu().detach().item(), ent_name)
                        )
                        index_set.add(idx_val)
        else:
            ent_name_result = []

        # Process observability mask results
        related_o11y_sorted_mask = []
        related_o11y_names = []
        for modal_type in self.meta_data['modal_types']:
            if self.hard_o11y_mask and self.hard_o11y_mask[modal_type].any():
                mask_vals, indices = torch.sort(
                    self.o11y_mask[modal_type][self.hard_o11y_mask[modal_type]].sigmoid(),
                    descending=True
                )
                # Get sorted mask values and corresponding names
                related_o11y_sorted_mask.extend(mask_vals.cpu().detach().tolist())
                o11y_names_filtered = np.array(self.meta_data['o11y_names'][modal_type])[
                    self.hard_o11y_mask[modal_type].cpu().detach()
                ]
                related_o11y_names.extend(o11y_names_filtered[indices.cpu().detach()].tolist())

        # Sort observability results
        o11y_name_result = sorted(
            zip(related_o11y_sorted_mask, related_o11y_names),
            reverse=True
        )

        # Clean up explainer state
        self.clean_explainer()

        return ent_name_result, o11y_name_result

    def explainer_loss(self, y_pred, y_true):
        """Calculate explainer loss (prediction loss + regularization)"""
        # Base prediction loss
        loss = self.criterion(y_pred, y_true)

        # Entity edge mask regularization
        if self.hard_ent_edge_mask is not None and self.hard_ent_edge_mask.any():
            assert self.ent_edge_mask is not None
            m = self.ent_edge_mask[self.hard_ent_edge_mask].sigmoid()
            # Size regularization (encourage sparse masks)
            ent_edge_reduce = getattr(torch, self.coeffs['ent_edge_reduction'])
            loss += self.coeffs['ent_edge_size'] * ent_edge_reduce(m)
            # Entropy regularization (encourage binary masks)
            entropy = -m * torch.log(m + self.coeffs['EPS']) - (1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss += self.coeffs['ent_edge_entropy'] * entropy.mean()

        # Observability mask regularization
        if self.hard_o11y_mask is not None:
            assert self.o11y_mask is not None
            modal_count = len(self.meta_data['modal_types'])
            for modal_type in self.meta_data['modal_types']:
                if self.hard_o11y_mask[modal_type].any():
                    m = self.o11y_mask[modal_type][self.hard_o11y_mask[modal_type]].sigmoid()
                    # Size regularization
                    o11y_reduce = getattr(torch, self.coeffs['o11y_reduction'])
                    loss += self.coeffs['o11y_size'] * o11y_reduce(m) / modal_count
                    # Entropy regularization
                    entropy = -m * torch.log(m + self.coeffs['EPS']) - (1 - m) * torch.log(1 - m + self.coeffs['EPS'])
                    loss += self.coeffs['o11y_entropy'] * entropy.mean() / modal_count

        return loss
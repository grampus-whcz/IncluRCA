import torch
import torch.nn as nn
from collections import OrderedDict

class SKAttention1D(nn.Module):
    """
    Selective Kernel Attention for 1D feature maps (B, C, L)
    """
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=8, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.kernels = kernels
        self.convs = nn.ModuleList()
        
        # Build multi-scale 1D conv branches
        for k in kernels:
            padding = k // 2
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv1d(channel, channel, kernel_size=k, padding=padding, groups=group)),
                    ('bn', nn.BatchNorm1d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        
        # Fuse: dimension reduction
        self.fc = nn.Linear(channel, self.d)
        
        # Branch-specific FCs to recover channel weights
        self.fcs = nn.ModuleList()
        for _ in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        """
        x: (B, C, L)
        """
        B, C, L = x.shape
        conv_outs = []

        # Multi-scale convolution
        for conv in self.convs:
            out = conv(x)  # (B, C, L)
            conv_outs.append(out)

        # Fuse: sum across scales
        U = sum(conv_outs)  # (B, C, L)

        # Global average pooling over L -> (B, C)
        S = U.mean(dim=-1)  # (B, C)

        # Dimension reduction
        Z = self.fc(S)  # (B, d)

        # Compute attention weights for each scale
        weights = []
        for fc in self.fcs:
            w = fc(Z)  # (B, C)
            w = w.view(B, C, 1)  # (B, C, 1) — broadcastable over L
            weights.append(w)
        
        scale_weights = torch.stack(weights, dim=0)  # (K, B, C, 1)
        scale_weights = self.softmax(scale_weights)  # (K, B, C, 1)

        # Stack features for weighting
        feats = torch.stack(conv_outs, dim=0)  # (K, B, C, L)

        # Select: weighted sum
        V = (scale_weights * feats).sum(dim=0)  # (B, C, L)

        return V

class FeatureIntegration(nn.Module):
    def __init__(self, param_dict, meta_data):
        super().__init__()
        self.device_marker = nn.Parameter(torch.empty(0))
        self.meta_data = meta_data

        self.se_attention_dict = nn.ModuleDict()
        self.ent_feature_align_dict = nn.ModuleDict()

        in_dim = param_dict['efi_in_dim']

        for ent_type in self.meta_data['ent_types']:
            all_ent_feature_length = 0
            for modal_type in self.meta_data['modal_types']:
                all_ent_feature_length += self.meta_data['max_ent_feature_num'][ent_type][modal_type]

            self.se_attention_dict[ent_type] = SKAttention1D(channel=in_dim)

            self.ent_feature_align_dict[ent_type] = nn.Linear(all_ent_feature_length * in_dim, param_dict['efi_out_dim'])

    def forward(self, batch_data):
        batch_size = batch_data['y'].shape[0]

        x_ent = []
        for ent_type in self.meta_data['ent_types']:
            for ent_index in range(self.meta_data['ent_type_index'][ent_type][0], self.meta_data['ent_type_index'][ent_type][1]):
                x = []
                for modal_type in self.meta_data['modal_types']:
                    feature_index_pair = self.meta_data['ent_features'][modal_type][ent_index][1]
                    modal_data = batch_data[f'x_{modal_type}'][:, feature_index_pair[0]:feature_index_pair[1], :]
                    padding = torch.zeros(batch_size, self.meta_data['max_ent_feature_num'][ent_type][modal_type] - modal_data.shape[1], modal_data.shape[2]).to(self.device_marker.device)
                    modal_data = torch.cat((modal_data, padding), 1)
                    x.append(modal_data)
                x = torch.cat(x, dim=1)  # (B, L, C)
                x = x.permute(0, 2, 1).contiguous()  # (B, C, L)
                x = self.se_attention_dict[ent_type](x)
                x = x.permute(0, 2, 1).contiguous()  # (B, L, C)
                x = x.view(batch_size, x.shape[1] * x.shape[2]).contiguous()
                x = self.ent_feature_align_dict[ent_type](x)
                x_ent.append(x)
        x_ent = torch.stack(x_ent, dim=1)
        batch_data['x_ent'] = x_ent
        return batch_data
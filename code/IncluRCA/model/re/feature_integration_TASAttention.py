import torch
import torch.nn as nn
import torch.nn.functional as F


class TrendAwareAttention(nn.Module):
    """
    Trend-aware Self-Attention from ASTGNN (TKDE 2021)
    Input:  (B, T, N, D)  where N = number of features per entity
    Output: (B, T, N, D)
    """
    def __init__(self, K, d, kernel_size, in_dim):
        super().__init__()
        assert in_dim == K * d, f"in_dim ({in_dim}) must equal K*d ({K}*{d})"
        self.d = d
        self.K = K
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1

        self.FC_v = nn.Linear(in_dim, in_dim)
        self.FC_out = nn.Linear(in_dim, in_dim)

        # Causal conv over time dimension (last dim after permute)
        self.cnn_q = nn.Conv2d(
            in_dim, in_dim,
            kernel_size=(1, kernel_size),
            padding=(0, self.padding),
            groups=in_dim
        )
        self.cnn_k = nn.Conv2d(
            in_dim, in_dim,
            kernel_size=(1, kernel_size),
            padding=(0, self.padding),
            groups=in_dim
        )
        self.norm_q = nn.BatchNorm2d(in_dim)
        self.norm_k = nn.BatchNorm2d(in_dim)

    def forward(self, X):
        # X: (B, T, N, D)
        B, T, N, D = X.shape
        if N == 0:
            return X  # should not happen if caller checks, but safe

        X_ = X.permute(0, 3, 2, 1)  # (B, D, N, T)

        # Apply causal convolution and remove extra padding
        q = self.norm_q(self.cnn_q(X_))[:, :, :, :-self.padding].permute(0, 3, 2, 1)  # (B, T, N, D)
        k = self.norm_k(self.cnn_k(X_))[:, :, :, :-self.padding].permute(0, 3, 2, 1)  # (B, T, N, D)
        v = self.FC_v(X)  # (B, T, N, D)

        # Multi-head split
        q = torch.cat(torch.split(q, self.d, dim=-1), dim=0)
        k = torch.cat(torch.split(k, self.d, dim=-1), dim=0)
        v = torch.cat(torch.split(v, self.d, dim=-1), dim=0)

        # Reshape for attention
        q = q.permute(0, 2, 1, 3)  # (B*K, N, T, d)
        k = k.permute(0, 2, 3, 1)  # (B*K, N, d, T)
        v = v.permute(0, 2, 1, 3)  # (B*K, N, T, d)

        attn = (q @ k) / (self.d ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).contiguous()
        out = torch.cat(torch.split(out, B, dim=0), dim=-1)  # (B, N, T, D)
        out = self.FC_out(out.permute(0, 2, 1, 3))  # (B, T, N, D)
        return out


class FeatureIntegration(nn.Module):
    def __init__(self, param_dict, meta_data):
        super().__init__()
        self.device_marker = nn.Parameter(torch.empty(0))
        self.meta_data = meta_data

        # Initialize ModuleDicts
        self.trend_attn_dict = nn.ModuleDict()
        self.ent_feature_align_dict = nn.ModuleDict()

        # Shared embedding for scalar → vector
        self.feature_embed = nn.Linear(1, param_dict['efi_in_dim'])

        in_dim = param_dict['efi_in_dim']
        efi_out_dim = param_dict['efi_out_dim']
        K = param_dict.get('trend_K', 8)
        d = param_dict.get('trend_d', 64)
        kernel_size = param_dict.get('trend_kernel_size', 3)

        assert in_dim == K * d, f"in_dim ({in_dim}) must equal K*d ({K}*{d})"

        # Precompute max total features per entity type
        self.F_total_dict = {}
        for ent_type in self.meta_data['ent_types']:
            total = 0
            for modal_type in self.meta_data['modal_types']:
                total += self.meta_data['max_ent_feature_num'][ent_type].get(modal_type, 0)
            self.F_total_dict[ent_type] = total

        # Build modules per entity type
        for ent_type in self.meta_data['ent_types']:
            F_total = self.F_total_dict[ent_type]
            self.trend_attn_dict[ent_type] = TrendAwareAttention(
                K=K, d=d, kernel_size=kernel_size, in_dim=in_dim
            )
            self.ent_feature_align_dict[ent_type] = nn.Linear(F_total * in_dim, efi_out_dim)

    def forward(self, batch_data):
        batch_size = batch_data['y'].shape[0]
        T = batch_data['x_metric'].shape[1]  # assume all modalities have same T

        x_ent = []
        for ent_type in self.meta_data['ent_types']:
            start_idx, end_idx = self.meta_data['ent_type_index'][ent_type]

            for ent_index in range(start_idx, end_idx):
                features_list = []
                total_feat_count = 0

                # Collect features from all modalities
                for modal_type in self.meta_data['modal_types']:
                    idx_range = None
                    for feat_info in self.meta_data['ent_features'][modal_type]:
                        if isinstance(feat_info, dict):
                            name_key = feat_info.get('name', '')
                            if name_key == f"{ent_type}-{ent_index+1}/{modal_type}":
                                idx_range = feat_info.get('idx')
                                break
                        elif isinstance(feat_info, tuple) and len(feat_info) == 2:
                            name_key, idx_tuple = feat_info
                            if name_key == f"{ent_type}-{ent_index+1}/{modal_type}":
                                idx_range = idx_tuple
                                break

                    if idx_range is None:
                        continue

                    # Extract data
                    x_modal = batch_data[f'x_{modal_type}']  # (B, T, F_modal)
                    start, end = idx_range
                    modal_data = x_modal[:, :, start:end]    # (B, T, f)

                    # Pad to max length for this (ent_type, modal_type)
                    max_len = self.meta_data['max_ent_feature_num'][ent_type].get(modal_type, 0)
                    current_len = modal_data.shape[2]
                    if current_len < max_len:
                        padding = torch.zeros(
                            batch_size, T, max_len - current_len,
                            device=self.device_marker.device,
                            dtype=modal_data.dtype
                        )
                        modal_data = torch.cat([modal_data, padding], dim=2)
                    elif current_len > max_len:
                        modal_data = modal_data[:, :, :max_len]

                    features_list.append(modal_data)
                    total_feat_count += modal_data.shape[2]

                # >>>>>>>>>> CRITICAL: Handle empty feature case <<<<<<<<<<
                if total_feat_count == 0:
                    # No features available → output zero vector
                    zero_vec = torch.zeros(
                        batch_size,
                        self.ent_feature_align_dict[ent_type].out_features,
                        device=self.device_marker.device
                    )
                    x_ent.append(zero_vec)
                    continue

                # Concatenate features
                x = torch.cat(features_list, dim=2)  # (B, T, F_total), F_total > 0

                # Embed each scalar into vector space
                x = x.unsqueeze(-1)                  # (B, T, F_total, 1)
                x = self.feature_embed(x)           # (B, T, F_total, in_dim)

                # Apply trend-aware attention only if T >= kernel_size
                if T >= self.trend_attn_dict[ent_type].kernel_size:
                    x = self.trend_attn_dict[ent_type](x)  # (B, T, F_total, in_dim)

                # Flatten and project to unified output dim
                x = x.view(batch_size, -1)  # (B, T * F_total * in_dim)
                x = self.ent_feature_align_dict[ent_type](x)  # (B, efi_out_dim)
                x_ent.append(x)

        x_ent = torch.stack(x_ent, dim=1)  # (B, num_entities, efi_out_dim)
        batch_data['x_ent'] = x_ent
        return batch_data
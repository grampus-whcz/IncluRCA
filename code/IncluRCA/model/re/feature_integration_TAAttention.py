import torch
import torch.nn as nn
import torch.nn.functional as F

class Trend_aware_attention(nn.Module):
    def __init__(self, K, d, kernel_size):
        super().__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_v = nn.Linear(D, D)
        self.FC = nn.Linear(D, D)
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.cnn_q = nn.Conv2d(D, D, (1, kernel_size), padding=(0, self.padding))
        self.cnn_k = nn.Conv2d(D, D, (1, kernel_size), padding=(0, self.padding))
        self.norm_q = nn.BatchNorm2d(D)
        self.norm_k = nn.BatchNorm2d(D)

    def forward(self, X):
        # X: (B, T, N, D)
        batch_size = X.shape[0]
        X_ = X.permute(0, 3, 2, 1)  # (B, D, N, T)

        query = self.norm_q(self.cnn_q(X_))[:, :, :, :-self.padding].permute(0, 3, 2, 1)
        key = self.norm_k(self.cnn_k(X_))[:, :, :, :-self.padding].permute(0, 3, 2, 1)
        value = self.FC_v(X)

        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)

        query = query.permute(0, 2, 1, 3)   # (B*K, N, T, d)
        key = key.permute(0, 2, 3, 1)       # (B*K, N, d, T)
        value = value.permute(0, 2, 1, 3)   # (B*K, N, T, d)

        attention = (query @ key) * (self.d ** -0.5)
        attention = F.softmax(attention, dim=-1)

        X_out = (attention @ value)
        X_out = torch.cat(torch.split(X_out, batch_size, dim=0), dim=-1)  # (B, N, T, D)
        X_out = self.FC(X_out)
        return X_out.permute(0, 2, 1, 3)  # (B, T, N, D)


class FeatureIntegration(nn.Module):
    def __init__(self, param_dict, meta_data):
        super().__init__()
        self.device_marker = nn.Parameter(torch.empty(0))
        self.meta_data = meta_data

        self.trend_attention_dict = nn.ModuleDict()
        self.ent_feature_align_dict = nn.ModuleDict()

        in_dim = param_dict['efi_in_dim']
        # Assume K = 4 and d = in_dim // 4, which requires that in_dim % 4 == 0
        K = 4
        d = in_dim // K
        assert in_dim % K == 0, f"in_dim {in_dim} must be divisible by K={K}"

        for ent_type in self.meta_data['ent_types']:
            all_ent_feature_length = 0
            for modal_type in self.meta_data['modal_types']:
                all_ent_feature_length += self.meta_data['max_ent_feature_num'][ent_type][modal_type]

            self.trend_attention_dict[ent_type] = Trend_aware_attention(K=K, d=d, kernel_size=3)

            self.ent_feature_align_dict[ent_type] = nn.Linear(all_ent_feature_length * in_dim, param_dict['efi_out_dim'])

    def forward(self, batch_data):
        batch_size = batch_data['y'].shape[0]
        x_ent = []

        for ent_type in self.meta_data['ent_types']:
            for ent_index in range(
                self.meta_data['ent_type_index'][ent_type][0],
                self.meta_data['ent_type_index'][ent_type][1]
            ):
                x_list = []
                for modal_type in self.meta_data['modal_types']:
                    feature_index_pair = self.meta_data['ent_features'][modal_type][ent_index][1]
                    modal_data = batch_data[f'x_{modal_type}'][:, feature_index_pair[0]:feature_index_pair[1], :]
                    pad_len = self.meta_data['max_ent_feature_num'][ent_type][modal_type] - modal_data.shape[1]
                    if pad_len > 0:
                        padding = torch.zeros(
                            batch_size, pad_len, modal_data.shape[2],
                            device=self.device_marker.device,
                            dtype=modal_data.dtype
                        )
                        modal_data = torch.cat((modal_data, padding), dim=1)
                    x_list.append(modal_data)
                x = torch.cat(x_list, dim=1)  # (B, L, C), L = total slots

                # Reshape to (B, T, N, D) for trend attention: T=L, N=1, D=C
                B, L, C = x.shape
                x = x.unsqueeze(2)  # (B, L, 1, C)

                # Apply trend-aware attention
                x = self.trend_attention_dict[ent_type](x)  # (B, L, 1, C)

                # Back to (B, L, C)
                x = x.squeeze(2)  # (B, L, C)

                # Flatten and align
                x = x.view(batch_size, -1)  # (B, L*C)
                x = self.ent_feature_align_dict[ent_type](x)  # (B, out_dim)
                x_ent.append(x)

        x_ent = torch.stack(x_ent, dim=1)  # (B, num_ent, out_dim)
        batch_data['x_ent'] = x_ent
        return batch_data
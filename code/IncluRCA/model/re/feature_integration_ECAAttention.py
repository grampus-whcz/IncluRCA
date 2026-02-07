import torch
import torch.nn as nn

class ECAAttention(nn.Module):
    def __init__(self, channel=None, gamma=2, b=1, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            if channel is None:
                raise ValueError("Either 'channel' or 'kernel_size' must be provided.")
            k = int(abs((torch.log2(torch.tensor(channel)) + b) / gamma))
            k = max(k, 3)
            if k % 2 == 0:
                k += 1
            kernel_size = k
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, L = x.shape
        y = self.avg_pool(x)  # (B, C, 1)
        y = y.squeeze(-1).unsqueeze(1)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = self.sigmoid(y).squeeze(1).unsqueeze(-1)  # (B, C, 1)
        return x * y.expand_as(x)

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

            self.se_attention_dict[ent_type] = ECAAttention(channel=in_dim)

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
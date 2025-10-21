import torch
import torch.nn as nn

class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.size()
        # Squeeze: (B,C,L)-->avg_pool-->(B,C,1)-->view-->(B,C)
        y = self.avg_pool(x.unsqueeze(-1)).view(B, C)  # (B, C)
        # Excitation: (B,C)-->fc-->(B,C)-->(B, C, 1)
        y = self.fc(y).view(B, C, 1)  # (B, C, 1)
        # scale: (B,C,L) * (B, C, 1) == (B,C,L)
        out = x * y  # (B, C, L)
        return out

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

            self.se_attention_dict[ent_type] = SEAttention(channel=in_dim)

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
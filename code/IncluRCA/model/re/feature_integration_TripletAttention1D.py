import torch
import torch.nn as nn

class BasicConv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ZPool1d(nn.Module):
    def forward(self, x):
        # x: (B, D1, D2)
        max_out = torch.max(x, dim=1, keepdim=True)[0]  # (B, 1, D2)
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, D2)
        return torch.cat([max_out, avg_out], dim=1)    # (B, 2, D2)

class AttentionGate1d(nn.Module):
    def __init__(self, kernel_size=7):
        super(AttentionGate1d, self).__init__()
        self.compress = ZPool1d()
        self.conv = BasicConv1d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # x: (B, D1, D2)
        x_compress = self.compress(x)      # (B, 2, D2)
        x_out = self.conv(x_compress)      # (B, 1, D2)
        scale = torch.sigmoid(x_out)       # (B, 1, D2)
        return x * scale                   # (B, D1, D2)

class TripletAttention1D(nn.Module):
    """
    Triplet Attention for 1D feature maps (B, C, L)
    Only two meaningful dimensions: C (channel) and L (length/sequence)
    We implement two branches:
      1. Compress over L → attend over C-L (permute to (B, L, C))
      2. Compress over C → attend over L-C (original (B, C, L))
    """
    def __init__(self, kernel_size=7):
        super(TripletAttention1D, self).__init__()
        self.branch1 = AttentionGate1d(kernel_size=kernel_size)  # operates on (B, L, C)
        self.branch2 = AttentionGate1d(kernel_size=kernel_size)  # operates on (B, C, L)

    def forward(self, x):
        """
        x: (B, C, L)
        """
        B, C, L = x.shape

        # Branch 1: model interaction between C and L by permuting to (B, L, C)
        x_perm = x.permute(0, 2, 1).contiguous()  # (B, L, C)
        out1 = self.branch1(x_perm)               # (B, L, C)
        out1 = out1.permute(0, 2, 1).contiguous() # (B, C, L)

        # Branch 2: model interaction by compressing over C (treat C as "height")
        out2 = self.branch2(x)                    # (B, C, L)

        # Average the two attention outputs
        out = 0.5 * (out1 + out2)
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

            self.se_attention_dict[ent_type] = TripletAttention1D(kernel_size=3)  # kernel_size = 5, 7 

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
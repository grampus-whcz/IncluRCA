import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConvSEAttention(nn.Module):
    """
    改进版通道注意力模块，支持：
    - 多种压缩方式（avg, max, conv, avg+max）
    - 激励部分使用多尺度1D卷积替代全连接，实现局部通道建模
    """
    def __init__(self, channel=512, reduction=16, squeeze_type='avg', excite_type='multi_conv'):
        super().__init__()
        self.squeeze_type = squeeze_type
        self.excite_type = excite_type
        self.channel = channel

        # ===== Squeeze 部分 =====
        if squeeze_type == 'avg':
            self.squeeze = lambda x: torch.mean(x, dim=-1, keepdim=True)  # (B, C, 1)
        elif squeeze_type == 'max':
            self.squeeze = lambda x: torch.max(x, dim=-1, keepdim=True)[0]
        elif squeeze_type == 'avg_max':
            self.squeeze_avg = lambda x: torch.mean(x, dim=-1, keepdim=True)
            self.squeeze_max = lambda x: torch.max(x, dim=-1, keepdim=True)[0]
        elif squeeze_type == 'conv':
            # 使用可学习的1D卷积压缩到1个位置
            self.squeeze_conv = nn.Conv1d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel)
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError(f"Unsupported squeeze_type: {squeeze_type}")

        # ===== Excitation 部分 =====
        if excite_type == 'fc':  # 原始SE
            self.excite = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid()
            )
        elif excite_type == 'conv':
            # 单尺度卷积
            self.excite = nn.Sequential(
                nn.Conv1d(channel, channel // reduction, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel // reduction, channel, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        elif excite_type == 'multi_conv':
            # 多尺度卷积：并行不同kernel_size的卷积
            self.conv1 = nn.Conv1d(channel, channel // reduction, kernel_size=3, padding=1, groups=1, bias=False)
            self.conv2 = nn.Conv1d(channel, channel // reduction, kernel_size=5, padding=2, groups=1, bias=False)
            self.conv3 = nn.Conv1d(channel, channel // reduction, kernel_size=7, padding=3, groups=1, bias=False)
            self.fuse = nn.Conv1d(3 * (channel // reduction), channel, kernel_size=1, bias=False)
            self.sigmoid = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported excite_type: {excite_type}")

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape

        # === Squeeze ===
        if self.squeeze_type == 'avg' or self.squeeze_type == 'max':
            y = self.squeeze(x)  # (B, C, 1)
        elif self.squeeze_type == 'avg_max':
            y_avg = self.squeeze_avg(x)
            y_max = self.squeeze_max(x)
            y = y_avg + y_max  # (B, C, 1)
        elif self.squeeze_type == 'conv':
            y = self.squeeze_conv(x)  # (B, C, L)
            y = self.pool(y)         # (B, C, 1)

        # === Excitation ===
        if self.excite_type == 'fc':
            y = y.view(B, C)  # (B, C)
            y = self.excite(y).view(B, C, 1)  # (B, C, 1)
        elif self.excite_type == 'conv':
            y = self.excite(y)  # (B, C, 1)
        elif self.excite_type == 'multi_conv':
            y1 = self.conv1(y.expand(-1, -1, L))  # expand to original L for conv
            y2 = self.conv2(y.expand(-1, -1, L))
            y3 = self.conv3(y.expand(-1, -1, L))
            y_cat = torch.cat([y1, y2, y3], dim=1)  # (B, 3*C//r, L)
            y = self.fuse(y_cat)  # (B, C, L)
            y = torch.mean(y, dim=-1, keepdim=True)  # global pool back to (B, C, 1)
            y = self.sigmoid(y)

        # === Scale ===
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
        
        squeeze_type = param_dict['squeeze_type']
        excite_type = param_dict['excite_type']
        
        print(f'FeatureIntegration using MultiScaleConvSEAttention with squeeze_type={squeeze_type}, excite_type={excite_type}')

        for ent_type in self.meta_data['ent_types']:
            all_ent_feature_length = 0
            for modal_type in self.meta_data['modal_types']:
                all_ent_feature_length += self.meta_data['max_ent_feature_num'][ent_type][modal_type]

            # self.se_attention_dict[ent_type] = SEAttention(channel=in_dim)
            # 在 FeatureIntegration.__init__ 中替换：
            self.se_attention_dict[ent_type] = MultiScaleConvSEAttention(
                channel=in_dim,
                reduction=32,
                squeeze_type = squeeze_type,      # 可选: 'avg', 'max', 'avg_max', 'conv'
                excite_type = excite_type     # 可选: 'fc', 'conv', 'multi_conv'
            )

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
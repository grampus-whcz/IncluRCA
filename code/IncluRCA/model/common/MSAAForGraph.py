import torch
import torch.nn as nn

class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [B, N, D]
        # 全局平均池化 over N (dim=1) → [B, D]
        avg_out = self.fc(x.mean(dim=1))          # [B, D]
        # 全局最大池化 over N (dim=1) → [B, D]
        max_out = self.fc(x.max(dim=1).values)    # [B, D]
        # 注意力权重
        att = self.sigmoid(avg_out + max_out).unsqueeze(1)  # [B, 1, D]
        return x * att  # [B, N, D]


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [B, N, D]
        avg_out = torch.mean(x, dim=2, keepdim=True)   # [B, N, 1]
        max_out, _ = torch.max(x, dim=2, keepdim=True) # [B, N, 1]
        concat = torch.cat([avg_out, max_out], dim=2)  # [B, N, 2]
        concat = concat.transpose(1, 2)                # [B, 2, N]
        att = self.conv(concat)                        # [B, 1, N]
        att = self.sigmoid(att).transpose(1, 2)        # [B, N, 1]
        return x * att  # [B, N, D]
    
class MSAAForGraph(nn.Module):
    def __init__(self, in_dim, out_dim, reduction=4):
        super().__init__()
        # 输入是 3 个 in_dim 拼起来 → 3 * in_dim
        self.down = nn.Linear(3 * in_dim, in_dim)
        # self.down = nn.Linear(2 * in_dim, in_dim)
        self.channel_att = ChannelAttention1D(in_dim, reduction)
        self.spatial_att = SpatialAttention1D()
        self.up = nn.Linear(in_dim, out_dim)

    def forward(self, x1, x2, x3):
        x_fused = torch.cat([x1, x2, x3], dim=-1)  # [B, N, 3 * in_dim]
        x_fused = self.down(x_fused)               # [B, N, in_dim]
        x_c = self.channel_att(x_fused)
        x_s = self.spatial_att(x_fused)
        x_out = self.up(x_c + x_s)                 # [B, N, out_dim]
        return x_out
    # def forward(self, x1, x2):
    #     x_fused = torch.cat([x1, x2], dim=-1)  # [B, N, 3 * in_dim]
    #     x_fused = self.down(x_fused)               # [B, N, in_dim]
    #     x_c = self.channel_att(x_fused)
    #     x_s = self.spatial_att(x_fused)
    #     x_out = self.up(x_c + x_s)                 # [B, N, out_dim]
    #     return x_out
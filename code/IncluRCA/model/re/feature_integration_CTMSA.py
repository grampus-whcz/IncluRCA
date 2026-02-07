import torch
import torch.nn as nn
from typing import Dict, Any, List

# ----------------------------
# 重构 TemporalAttention：支持动态窗口
# 因果时间注意⼒ (CT-MSA, AAAI2023)
# ----------------------------

class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=2, qkv_bias=False, qk_scale=None, dropout=0., causal=True):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."
        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, window_size: int = 1):
        B_prev, T_prev, C_prev = x.shape
        if window_size > 0 and window_size < T_prev:
            # 分块处理（非重叠窗口）
            pad_len = (window_size - T_prev % window_size) % window_size
            if pad_len > 0:
                x = torch.cat([x, torch.zeros_like(x[:, :pad_len])], dim=1)
            B, T, C = x.shape
            x = x.view(B, T // window_size, window_size, C)
        else:
            window_size = T_prev
            B, T, C = x.shape
            x = x.unsqueeze(1)  # (B, 1, T, C)

        B, N_win, T, C = x.shape
        x = x.view(-1, T, C)  # (B*N_win, T, C)

        qkv = self.qkv(x).reshape(-1, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.causal:
            mask = torch.tril(torch.ones(T, T, device=x.device))
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(-1, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.view(B, N_win, T, C)
        x = x.reshape(B, N_win * T, C)

        if window_size < T_prev:
            x = x[:, :T_prev]  # 去掉 padding

        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class CT_MSA(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, num_time_max, dropout=0.):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_time_max, dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim=dim, heads=heads, dropout=dropout, causal=True),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, window_size: int):
        # x: (B, T, C)
        B, T, C = x.shape
        x = x + self.pos_embedding[:, :T, :]  # 自动截断

        for attn, ff in self.layers:
            x = attn(x, window_size=window_size) + x
            x = ff(x) + x
        return x


# ----------------------------
# 增强版 FeatureIntegration
# ----------------------------

class FeatureIntegration(nn.Module):
    def __init__(
        self,
        param_dict: Dict[str, Any],
        meta_data: Dict[str, Any],
        shared_ct_msa: bool = False,
        use_entity_type_embed: bool = True,
        use_modal_embed: bool = True,
        window_config: List[int] = [4, 8, 16]  # 动态窗口策略：按块索引选择
    ):
        super().__init__()
        self.device_marker = nn.Parameter(torch.empty(0), requires_grad=False)
        self.meta_data = meta_data
        self.shared_ct_msa = shared_ct_msa
        self.use_entity_type_embed = use_entity_type_embed
        self.use_modal_embed = use_modal_embed
        self.window_config = window_config  # e.g., block0 → 4, block1 → 8, ...

        in_dim = param_dict['efi_in_dim']
        out_dim = param_dict['efi_out_dim']
        heads = param_dict.get('efi_ct_heads', 4)
        mlp_dim = param_dict.get('efi_ct_mlp_dim', in_dim * 2)
        dropout = param_dict.get('efi_dropout', 0.1)
        depth = param_dict.get('efi_ct_depth', 1)
        blocks = param_dict.get('efi_ct_blocks', len(window_config))

        self.in_dim = in_dim
        self.blocks = blocks

        # 实体类型嵌入
        if self.use_entity_type_embed:
            self.ent_type_to_id = {et: i for i, et in enumerate(meta_data['ent_types'])}
            self.ent_type_embed = nn.Embedding(len(meta_data['ent_types']), in_dim)

        # 模态嵌入
        if self.use_modal_embed:
            self.modal_to_id = {mt: i for i, mt in enumerate(meta_data['modal_types'])}
            self.modal_embed = nn.Embedding(len(meta_data['modal_types']), in_dim)

        # 构建 CT-MSA 模块
        if self.shared_ct_msa:
            max_total_len = max(
                sum(meta_data['max_ent_feature_num'][et][mt] for mt in meta_data['modal_types'])
                for et in meta_data['ent_types']
            )
            self.shared_ct_modules = nn.ModuleList()
            for b in range(blocks):
                self.shared_ct_modules.append(
                    CT_MSA(
                        dim=in_dim,
                        depth=depth,
                        heads=heads,
                        mlp_dim=mlp_dim,
                        num_time_max=max_total_len,
                        dropout=dropout
                    )
                )
        else:
            self.ct_msa_dict = nn.ModuleDict()

        self.ent_feature_align_dict = nn.ModuleDict()

        for ent_type in meta_data['ent_types']:
            total_feat_len = sum(
                meta_data['max_ent_feature_num'][ent_type][mt]
                for mt in meta_data['modal_types']
            )

            if not self.shared_ct_msa:
                ct_modules = nn.ModuleList()
                for b in range(blocks):
                    ct_modules.append(
                        CT_MSA(
                            dim=in_dim,
                            depth=depth,
                            heads=heads,
                            mlp_dim=mlp_dim,
                            num_time_max=total_feat_len,
                            dropout=dropout
                        )
                    )
                self.ct_msa_dict[ent_type] = ct_modules

            self.ent_feature_align_dict[ent_type] = nn.Linear(total_feat_len * in_dim, out_dim)

    def _get_window_size(self, seq_len: int, block_idx: int) -> int:
        """动态窗口策略"""
        if block_idx >= len(self.window_config):
            base_win = self.window_config[-1]
        else:
            base_win = self.window_config[block_idx]
        # 窗口不超过序列长度
        return min(base_win, seq_len)

    def forward(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = batch_data['y'].shape[0]
        x_ent = []

        for ent_type in self.meta_data['ent_types']:
            ent_start, ent_end = self.meta_data['ent_type_index'][ent_type]

            for ent_index in range(ent_start, ent_end):
                x_modal_list = []
                total_feat_len = 0

                for modal_type in self.meta_data['modal_types']:
                    feat_start, feat_end = self.meta_data['ent_features'][modal_type][ent_index][1]
                    modal_data = batch_data[f'x_{modal_type}'][:, feat_start:feat_end, :]  # (B, T_m, C)
                    T_m = modal_data.shape[1]
                    max_T_m = self.meta_data['max_ent_feature_num'][ent_type][modal_type]

                    # 补零到 max
                    if T_m < max_T_m:
                        pad = torch.zeros(
                            batch_size, max_T_m - T_m, modal_data.shape[2],
                            device=self.device_marker.device
                        )
                        modal_data = torch.cat([modal_data, pad], dim=1)
                    elif T_m > max_T_m:
                        modal_data = modal_data[:, :max_T_m, :]  # 截断（安全兜底）

                    # 注入模态嵌入
                    if self.use_modal_embed:
                        modal_id = self.modal_to_id[modal_type]
                        mod_embed = self.modal_embed.weight[modal_id]  # (C,)
                        modal_data = modal_data + mod_embed  # 广播

                    x_modal_list.append(modal_data)
                    total_feat_len += max_T_m

                x = torch.cat(x_modal_list, dim=1)  # (B, T, C)

                # 注入实体类型嵌入
                if self.use_entity_type_embed:
                    ent_type_id = self.ent_type_to_id[ent_type]
                    type_embed = self.ent_type_embed.weight[ent_type_id]  # (C,)
                    x = x + type_embed

                # 应用 CT-MSA（仅当 T > 1）
                if total_feat_len > 1:
                    for b in range(self.blocks):
                        window_size = self._get_window_size(total_feat_len, b)
                        if self.shared_ct_msa:
                            x = self.shared_ct_modules[b](x, window_size=window_size)
                        else:
                            x = self.ct_msa_dict[ent_type][b](x, window_size=window_size)

                x = x.contiguous().view(batch_size, -1)  # (B, T*C)
                x = self.ent_feature_align_dict[ent_type](x)  # (B, out_dim)
                x_ent.append(x)

        x_ent = torch.stack(x_ent, dim=1)  # (B, num_entities, out_dim)
        batch_data['x_ent'] = x_ent
        return batch_data
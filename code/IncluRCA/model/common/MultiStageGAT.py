# IncluRCA/model/common/MultiStageGAT.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import MessagePassing


class MultiStageGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads, dropout,
                 GAT_name3='GATv2Conv', GAT_name4='GATv2Conv', GAT_name5='GATv2Conv',
                 activ_fun3='elu', activ_fun4='elu', activ_fun5='elu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.gat_out_dim = hidden_dim * heads

        gat_class3 = getattr(gnn, GAT_name3)
        gat_class4 = getattr(gnn, GAT_name4)
        gat_class5 = getattr(gnn, GAT_name5)

        self.gat3 = gat_class3(in_dim, hidden_dim, heads=heads, dropout=dropout, add_self_loops=False)
        self.gat4 = gat_class4(self.gat_out_dim, hidden_dim, heads=heads, dropout=dropout, add_self_loops=False)
        self.gat5 = gat_class5(self.gat_out_dim, hidden_dim, heads=heads, dropout=dropout, add_self_loops=False)

        self.activ_fun3 = getattr(F, activ_fun3)
        self.activ_fun4 = getattr(F, activ_fun4)
        self.activ_fun5 = getattr(F, activ_fun5)

    def forward(self, x, edge_index):
        # x: [B*N, D]
        x1 = self.activ_fun3(self.gat3(x, edge_index))
        x2 = self.activ_fun4(self.gat4(x1, edge_index))
        x3 = self.activ_fun5(self.gat5(x2, edge_index))
        return x1, x2, x3  # each: [B*N, hidden_dim * heads]
        # return x1, x2

    def get_gat_modules(self):
        """返回所有 GAT 层，用于 Explainer 注入 edge mask"""
        return [self.gat3, self.gat4, self.gat5]
        # return [self.gat3, self.gat4]
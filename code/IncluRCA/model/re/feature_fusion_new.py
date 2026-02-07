import torch
import torch.nn as nn
import torch.nn.functional as F
from IncluRCA.util.ent_batch_graph import EntBatchGraph
import torch_geometric.nn as gnn



    

# class FeatureFusion(nn.Module):
#     def __init__(self, param_dict, meta_data):
#         super().__init__()
#         self.device_marker = nn.Parameter(torch.empty(0))
#         self.meta_data = meta_data

#         in_dim = param_dict['eff_in_dim']  # e.g., 512
#         hidden_dim = param_dict.get('eff_hidden_dim', 64)  # e.g., 64
#         final_out_dim = param_dict['eff_GAT_out_channels']  # e.g., 256
#         heads = param_dict['eff_GAT_heads']  # e.g., 4
#         dropout = param_dict['eff_GAT_dropout']       

#         gat_out_dim = hidden_dim * heads  # e.g., 64 * 4 = 256
        
#         print(f"GAT in_dim: {in_dim}, hidden_dim: {hidden_dim}, heads: {heads}, gat_out_dim: {gat_out_dim}")
        
#         GAT_name3=param_dict['GAT_name3']
#         GAT_name4=param_dict['GAT_name4']
#         GAT_name5=param_dict['GAT_name5']
#         activ_fun3=param_dict['activ_fun3']
#         activ_fun4=param_dict['activ_fun4']
#         activ_fun5=param_dict['activ_fun5']
#         print("---- MSAA fusion ----")
#         print("GAT_name3: ", GAT_name3)
#         print("GAT_name4: ", GAT_name4)
#         print("GAT_name5: ", GAT_name5)
#         print("activ_fun3: ", activ_fun3)
#         print("activ_fun4: ", activ_fun4)
#         print("activ_fun5: ", activ_fun5)
        
#         gat_class3 = getattr(gnn, GAT_name3)
#         gat_class4 = getattr(gnn, GAT_name4)
#         gat_class5 = getattr(gnn, GAT_name5)
#         self.activ_fun3 = activ_fun3
#         self.activ_fun4 = activ_fun4
#         self.activ_fun5 = activ_fun5

#         self.gat3 = gat_class3(in_dim, hidden_dim, heads=heads, dropout=dropout, add_self_loops=False)
#         self.gat4 = gat_class4(gat_out_dim, hidden_dim, heads=heads, dropout=dropout, add_self_loops=False)
#         self.gat5 = gat_class5(gat_out_dim, hidden_dim, heads=heads, dropout=dropout, add_self_loops=False)

#         # 融合后输出维度为 final_out_dim
#         self.msaa = MSAAForGraph(in_dim=gat_out_dim, out_dim=final_out_dim)

#         # ❌ 删除 linear_dict —— 分类由 FaultClassifier 完成

#     def forward(self, batch_data):
#         ent_batch_graph = EntBatchGraph(batch_data, self.meta_data).to(self.device_marker.device)
#         x = ent_batch_graph.x['re']  # [B, N, D]
#         edge_index = ent_batch_graph.edge_index

#         B, N, D = x.shape
#         x_flat = x.view(B * N, D)
        
#         # GAT layers
#         gat_class3 = getattr(F, self.activ_fun3)
#         gat_class4 = getattr(F, self.activ_fun4)
#         gat_class5 = getattr(F, self.activ_fun5)
        
#         x1_flat = gat_class3(self.gat3(x_flat, edge_index))   # [B*N, hidden*heads]
#         x2_flat = gat_class4(self.gat4(x1_flat, edge_index))  # [B*N, hidden*heads]
#         x3_flat = gat_class5(self.gat5(x2_flat, edge_index))  # [B*N, hidden*heads]

#         x1 = x1_flat.view(B, N, -1)  # [B, N, gat_out_dim]
#         x2 = x2_flat.view(B, N, -1)
#         x3 = x3_flat.view(B, N, -1)

#         # 融合特征：[B, N, final_out_dim]
#         x_fused = self.msaa(x1, x2, x3)

#         # ✅ 直接返回融合后的张量，不进行分类
#         return x_fused  # Shape: [B, N, final_out_dim]


# IncluRCA/model/common/FeatureFusion.py (新版本)

import torch
import torch.nn as nn
from IncluRCA.util.ent_batch_graph import EntBatchGraph
from IncluRCA.model.common.MultiStageGAT import MultiStageGAT
from IncluRCA.model.common.MSAAForGraph import MSAAForGraph


class FeatureFusion(nn.Module):
    def __init__(self, param_dict, meta_data):
        super().__init__()
        self.device_marker = nn.Parameter(torch.empty(0))
        self.meta_data = meta_data

        in_dim = param_dict['eff_in_dim']
        hidden_dim = param_dict.get('eff_hidden_dim', 64)
        final_out_dim = param_dict['eff_GAT_out_channels']
        heads = param_dict['eff_GAT_heads']
        dropout = param_dict['eff_GAT_dropout']

        self.GAT_net = MultiStageGAT(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
            GAT_name3=param_dict['GAT_name3'],
            GAT_name4=param_dict['GAT_name4'],
            GAT_name5=param_dict['GAT_name5'],
            activ_fun3=param_dict['activ_fun3'],
            activ_fun4=param_dict['activ_fun4'],
            activ_fun5=param_dict['activ_fun5']
        )

        gat_out_dim = hidden_dim * heads
        
        print(f"GAT in_dim: {in_dim}, hidden_dim: {hidden_dim}, heads: {heads}, gat_out_dim: {gat_out_dim}")
        print("---- MSAA fusion ----")
        print("GAT_name3: ", param_dict['GAT_name3'])
        print("GAT_name4: ", param_dict['GAT_name4'])
        print("GAT_name5: ", param_dict['GAT_name5'])
        print("activ_fun3: ", param_dict['activ_fun3'])
        print("activ_fun4: ", param_dict['activ_fun4'])
        print("activ_fun5: ", param_dict['activ_fun5'])
        
        self.msaa = MSAAForGraph(in_dim=gat_out_dim, out_dim=final_out_dim)

    def forward(self, batch_data):
        ent_batch_graph = EntBatchGraph(batch_data, self.meta_data).to(self.device_marker.device)
        x = ent_batch_graph.x['re']  # [B, N, D]
        edge_index = ent_batch_graph.edge_index

        B, N, D = x.shape
        x_flat = x.view(B * N, D)

        x1_flat, x2_flat, x3_flat = self.GAT_net(x_flat, edge_index)
        # x1_flat, x2_flat = self.GAT_net(x_flat, edge_index)

        x1 = x1_flat.view(B, N, -1)
        x2 = x2_flat.view(B, N, -1)
        x3 = x3_flat.view(B, N, -1)

        x_fused = self.msaa(x1, x2, x3)
        # x_fused = self.msaa(x1, x2)
        return x_fused  # [B, N, final_out_dim]
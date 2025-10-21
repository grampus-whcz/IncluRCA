import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F


class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout, GAT_name1='GATv2Conv', GAT_name2='GATv2Conv', activ_fun1='elu', activ_fun2='elu'):
        super().__init__()
        
        print("----GATNet----")
        print("GAT_name1: ", GAT_name1)
        print("GAT_name2: ", GAT_name2)
        print("activ_fun1: ", activ_fun1)
        print("activ_fun2: ", activ_fun2)
        
        gat_class1 = getattr(gnn, GAT_name1)
        gat_class2 = getattr(gnn, GAT_name2)
        self.activ_fun1 = activ_fun1
        self.activ_fun2 = activ_fun2
        
        self.conv1 = gat_class1(in_channels=in_channels,    # GATConv, GATv2Conv
                                   out_channels=out_channels, # SuperGATConv, GATConv, GATv2Conv
                                   heads=heads,
                                   dropout=dropout,
                                   add_self_loops=False)
        self.conv2 = gat_class2(in_channels=out_channels * heads,
                                   out_channels=int(out_channels / heads),
                                   heads=heads,
                                   dropout=dropout,
                                   add_self_loops=False)
        self.dropout_ratio = dropout

    def forward(self, x, edge_index):
        
        gat_class1 = getattr(F, self.activ_fun1)
        gat_class2 = getattr(F, self.activ_fun2)
        
        batch_size = x.shape[0]
        x = x.view(x.shape[0] * x.shape[1], x.shape[2]).contiguous() # elu, hardtanh
        x = gat_class1(self.conv1(x, edge_index)) # hardtanh, mish, hardswish, silu, tanh, hardsigmoid, rrelu, leaky_relu, celu, selu, elu
        x = gat_class2(self.conv2(x, edge_index))
        x = x.view(batch_size, int(x.shape[0] / batch_size), x.shape[1]).contiguous()
        return x

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

# 定义 GAT 模型
class GAT(nn.Module):
    # def __init__(self, in_dim, hidden_dim, out_dim, num_heads=1, dropout=0.1):
    def __init__(self, configs):
        super(GAT, self).__init__()
        self.gat1 = GATConv(configs.in_dim, configs.hidden_dim, heads=configs.num_heads, dropout=configs.dropout, concat=True)
        self.gat2 = GATConv(configs.hidden_dim * configs.num_heads, configs.out_dim, heads=1, dropout=configs.dropout, concat=False)
        self.dropout = nn.Dropout(configs.dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        # 注意：GATConv 可以直接接受 edge_attr 作为边权重
        x = self.gat1(x, edge_index, edge_attr)
        x = self.dropout(self.relu(x))
        x = self.gat2(x, edge_index, edge_attr)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.LeakyReLU()  # 使用 ReLU 激活函数

    def forward(self, X, adj):
        """
        Args:
            X: [batch_size, num_nodes, in_features]
            adj: [batch_size, num_nodes, num_nodes]
        Returns:
            H_out: [batch_size, num_nodes, out_features]
        """
        batch_size, num_nodes, in_features = X.size()
        # 聚合邻居特征
        H_agg = torch.einsum("bij,bjk->bik", adj, X)  # 对应 A * H
        # 线性变换 + 激活函数
        H_out = self.linear(H_agg)
        H_out = self.activation(H_out)  # 对应 A * H * W + 激活函数
        return H_out

class GNNTimeModel(nn.Module):
    def __init__(self, configs):
        super(GNNTimeModel, self).__init__()
        self.num_time_steps = configs.gnn_in_timestamps
        self.num_variables = configs.input_channels
        self.activation = nn.ReLU()

        # 单头 GNN 层
        self.gnn_layer = GNNLayer(in_features=configs.gnn_in_features, out_features=configs.gnn_out_features)

        # 可学习邻接矩阵
        self.learnable_adj = nn.Parameter(torch.randn(self.num_time_steps, self.num_variables, self.num_variables))  # 可学习的邻接矩阵
        self.bn = nn.LayerNorm(configs.gnn_in_timestamps)

    def forward(self, X):
        """
        Args:
            X: [batch_size, num_variables, num_time_steps]
        Returns:
            output: [batch_size, out_features]
        """
        batch_size, num_variables, num_time_steps = X.size()
        X = X.permute(0, 2, 1).unsqueeze(-1)  # [batch_size, num_time_steps, num_variables, 1]

        gnn_outputs = []
        for t in range(num_time_steps):
            X_t = X[:, t, :, :]  # [batch_size, num_variables, 1]

            # 将学习到的邻接矩阵直接作为权重矩阵
            adj_t = torch.sigmoid(self.learnable_adj[t])  # [num_variables, num_variables]
            eye = torch.eye(adj_t.size(0), device=adj_t.device)  # 生成单位矩阵
            adj_t = adj_t * (1 - eye) + eye  # 替换对角线为 1


            # 扩展到 batch_size
            adj_t = adj_t.expand(batch_size, -1, -1)  # [batch_size, num_variables, num_variables]

            # 单头 GNN 层计算
            H_t = self.gnn_layer(X_t, adj_t)  # [batch_size, num_variables, gnn_features]
            gnn_outputs.append(H_t)

        # 拼接所有时间步
        gnn_outputs = torch.stack(gnn_outputs, dim=1)  # [batch_size, num_time_steps, num_variables, gnn_features]
        gnn_outputs = gnn_outputs.view(batch_size, num_time_steps,-1)
        # gnn_outputs = gnn_outputs.permute(0, 2, 1)
        # gnn_outputs = self.bn(gnn_outputs)
        # gnn_outputs = gnn_outputs.permute(0, 2, 1)
        return gnn_outputs

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.ReLU()  # 使用 ReLU 激活函数

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
        H_out = self.activation(self.linear(H_agg))  # 对应 A * H * W + 激活函数
        return H_out

class GNNTimeModel(nn.Module):
    def __init__(self, configs):
        super(GNNTimeModel, self).__init__()
        self.num_time_steps = configs.sequence_len
        self.num_variables = configs.input_channels

        # 单头 GNN 层
        self.gnn_layer = GNNLayer(in_features=configs.gnn_in_features, out_features=configs.gnn_out_features)

        # 可学习邻接矩阵
        self.learnable_adj = nn.Parameter(torch.randn(self.num_time_steps, self.num_variables, self.num_variables))  # 可学习的邻接矩阵

        # LSTM 层
        # self.lstm = nn.LSTM(input_size=self.num_variables * configs.gnn_out_features,
        #                     hidden_size=configs.lstm_hidden_size,
        #                     batch_first=True)

        # 全连接层
        # self.fc = nn.Linear(configs.lstm_hidden_size, configs.lstm_out_features)

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
        gnn_outputs = gnn_outputs.view(batch_size, num_time_steps, -1)  # Flatten nodes

        # LSTM 时间序列建模
        # lstm_out, _ = self.lstm(gnn_outputs)  # [batch_size, num_time_steps, lstm_hidden_size]

        # 最后时间步的隐藏状态用于预测
        # output = self.fc(lstm_out[:, -1, :])  # [batch_size, out_features]
        return gnn_outputs

# # 示例用法
# batch_size = 32
# num_variables = 9
# num_time_steps = 128
# gnn_features = 16
# lstm_hidden_size = 64
# out_features = 128
#
# # 输入数据
# X = torch.randn(batch_size, num_variables, num_time_steps)  # [batch_size, num_variables, num_time_steps]
#
# # 模型初始化
# model = GNNTimeModel(num_variables=num_variables,
#                                 num_time_steps=num_time_steps,
#                                 gnn_features=gnn_features,
#                                 lstm_hidden_size=lstm_hidden_size,
#                                 out_features=out_features)
#
# # 前向传播
# output = model(X)
# print("Output shape:", output.shape)  # [batch_size, out_features]

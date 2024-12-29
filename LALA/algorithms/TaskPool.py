import torch
import torch.nn.functional as F
from torch.optim import SGD

class TaskPool:
    def __init__(self, configs, device):
        self.pool_size = configs.pool_size
        self.matrix_size = configs.input_channels
        self.device = device
        self.lr = 0.1

        # 初始化矩阵池，值在 [0, 1] 之间
        self.pool = torch.rand(self.pool_size, self.matrix_size, self.matrix_size, device=self.device, requires_grad=True)

        # 优化器，用于更新池中矩阵
        self.optimizer = SGD([self.pool], lr=self.lr)

    def find_closest_matrix(self, A_dynamic):
        """
        找到与 A_dynamic 最接近的矩阵的索引和相似度。
        """
        similarities = []
        for A in self.pool:
            similarity = torch.norm(A_dynamic - A, p="fro")  # Frobenius norm
            similarities.append(similarity)

        closest_idx = torch.argmin(torch.stack(similarities))  # 注意需要 stack 成 tensor
        closest_matrix = self.pool[closest_idx]  # 不需要 .detach()
        return closest_idx, closest_matrix

    def compute_loss(self, A_dynamic, closest_idx):
        """
        计算与最近邻矩阵的 MSE 损失。
        """
        A_closest = self.pool[closest_idx]  # 从池中选择最近的矩阵
        loss = F.mse_loss(A_closest, A_dynamic)  # 使用 MSE 计算损失
        return loss

    def update_matrix(self, closest_idx, A_dynamic):
        """
        根据损失单独更新 TaskPool 中的矩阵。
        """
        self.optimizer.zero_grad()  # 清除梯度
        A_closest = self.pool[closest_idx]
        loss = F.mse_loss(A_closest, A_dynamic)  # 使用 MSE 计算损失
        loss.backward()  # 对 TaskPool 中的参数进行反向传播
        self.optimizer.step()  # 使用优化器更新池中参数
        return loss.item()

    def __repr__(self):
        return f"Matrix Pool:\n{self.pool.data}"

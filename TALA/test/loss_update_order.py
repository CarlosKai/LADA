import torch
import torch.nn as nn
import torch.optim as optim

# 定义 A 网络
class ANetwork(nn.Module):
    def __init__(self):
        super(ANetwork, self).__init__()
        self.fc = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(20)

    def forward(self, x):
        x = self.fc(x)
        y = self.bn(x).clone()
        return y

# 定义 B 网络
class BNetwork(nn.Module):
    def __init__(self):
        super(BNetwork, self).__init__()
        self.fc = nn.Linear(20, 30)

    def forward(self, x):
        return self.fc(x)

# 定义 C 网络
class CNetwork(nn.Module):
    def __init__(self):
        super(CNetwork, self).__init__()
        self.fc = nn.Linear(20, 30)

    def forward(self, x):
        return self.fc(x)

# 初始化网络
A = ANetwork()
B = BNetwork()
C = CNetwork()

# 损失函数
loss_fn = nn.MSELoss()

# 优化器
optimizer_A = optim.Adam(A.parameters(), lr=0.001)  # 用于更新 A 网络
optimizer_B = optim.Adam(B.parameters(), lr=0.001)  # 用于更新 B 网络
optimizer_C = optim.Adam(C.parameters(), lr=0.001)  # 用于更新 C 网络

# 模拟输入
x = torch.randn(32, 10)  # 输入
target_b = torch.randn(32, 30)  # B 的目标
target_c = torch.randn(32, 30)  # C 的目标

# 前向传播
A_out = A(x).clone()         # 保证 A 的输出不被修改
B_out = B(A_out.clone())     # 对 B 的输入使用 clone
C_out = C(A_out.clone())     # 对 C 的输入使用 clone

# 计算损失
loss_B = loss_fn(B_out, target_b)  # B 的损失
loss_C = loss_fn(C_out, target_c)  # C 的损失

# ---- 更新 B 网络和 A 网络 ----
optimizer_B.zero_grad()
optimizer_A.zero_grad()  # 确保 A 的梯度被清零
loss_B.backward(retain_graph=True)  # 计算损失 B 的梯度
optimizer_B.step()  # 更新 B 的参数
# optimizer_A.step()  # 更新 A 的参数

# ---- 更新 C 网络和 A 网络 ----
optimizer_C.zero_grad()
# optimizer_A.zero_grad()  # 确保 A 的梯度被清零
loss_C.backward()  # 计算损失 C 的梯度
optimizer_C.step()  # 更新 C 的参数
optimizer_A.step()  # 更新 A 的参数

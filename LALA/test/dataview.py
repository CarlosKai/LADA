import torch

file_path = '../../data/WISDM/train_1.pt'

# 尝试加载 PyTorch 的 .pt 文件
try:
    data = torch.load(file_path)  # 加载文件
    print(type(data))  # 检查数据类型
    print(data['samples'].shape)  # 打印数据内容
    print(data['labels'].shape)
except Exception as e:
    print(f"Error reading the file: {e}")

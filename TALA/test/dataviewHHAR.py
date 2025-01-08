import torch

file_path = '../../data/HHAR_SA/train_7.pt'

# 尝试加载 PyTorch 的 .pt 文件
try:
    data = torch.load(file_path)  # 加载文件
    print(type(data))  # 检查数据类型
    print(data['samples'].shape)  # 打印数据内容
    print(data['samples'][0])
    print(data['labels'].shape)
except Exception as e:
    print(f"Error reading the file: {e}")


# import torch
# import matplotlib.pyplot as plt
# import os
#
# # 创建保存图像的文件夹（如果不存在）
# output_dir = "./image"
# os.makedirs(output_dir, exist_ok=True)
#
# # 加载数据
# file_path = '../../data/HHAR_SA/train_1.pt'  # 替换为你的文件路径
# data = torch.load(file_path)
# samples = data['samples']
# labels = data['labels']
#
# # 确认数据形状
# print(f"Samples shape: {samples.shape}")
#
# # 绘制前100条数据
# for i in range(min(100, len(samples))):  # 确保不会超过样本数量
#     sample = samples[i]
#     label = labels[i]
#
#     # 创建图像
#     plt.figure(figsize=(7, 4))
#     plt.plot(sample[:80, 0], label='Z axis', linestyle='-', color='orange', linewidth=4.0)
#     plt.plot(sample[:80, 1], label='X axis', linestyle='--', color='green', linewidth=4.0)
#     plt.plot(sample[:80, 2], label='Y axis', linestyle='-.', color='blue', linewidth=4.0)
#
#     ax = plt.gca()
#     ax.set_xticks([])  # 去除x轴刻度
#     ax.set_yticks([])  # 去除y轴刻度
#     ax.spines['top'].set_visible(True)  # 顶部边框
#     ax.spines['right'].set_visible(True)  # 右侧边框
#     ax.spines['left'].set_visible(True)  # 左侧边框
#     ax.spines['bottom'].set_visible(True)  # 底部边框
#     ax.spines['top'].set_linewidth(2)  # 顶部边框宽度
#     ax.spines['right'].set_linewidth(2)  # 右侧边框宽度
#     ax.spines['left'].set_linewidth(2)  # 左侧边框宽度
#     ax.spines['bottom'].set_linewidth(2)  # 底部边框宽度
#
#     # 保存为矢量图 (SVG)，文件名只包含样本编号
#     plt.savefig(f"{output_dir}/sample_{i+1}_{label}.svg", format='svg', bbox_inches='tight', pad_inches=0)
#     plt.close()
#
# print("Plots successfully generated and saved as vector graphics.")
#
#

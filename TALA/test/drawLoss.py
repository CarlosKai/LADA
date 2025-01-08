import re
import matplotlib.pyplot as plt


# 定义读取文本并解析的函数
def parse_loss_data(file_path):
    # 初始化存储每个loss值的字典
    losses = {
        'Epoch': [],
        'Src_cls_loss': [],
        'Trg_cls_loss': [],
        'Domain_loss': [],
        'Src_task_loss': [],
        'class_feature_and_domain_loss': [],
        'Trg_MMD_loss': [],
        'Contrast_loss': [],
        'Contrast_loss_cross': [],
        'Contrast_loss_intra': [],
        'Total_loss': []
    }

    # 打开文件并逐行解析
    with open(file_path, 'r') as f:
        for line in f:
            # 提取Epoch信息
            if '[Epoch :' in line:
                epoch = int(re.search(r'\[Epoch : (\d+)/', line).group(1))
                losses['Epoch'].append(epoch)
            # 提取每个loss信息
            for key in losses.keys():
                if key in line and key != 'Epoch':
                    match = re.search(rf'{key}\s*:\s*([\d\.]+)', line)
                    if match:  # 确保匹配到值
                        value = float(match.group(1))
                        losses[key].append(value)

    # 确保每个key的长度一致
    for key in losses.keys():
        if len(losses[key]) < len(losses['Epoch']):
            losses[key].extend([None] * (len(losses['Epoch']) - len(losses[key])))

    return losses


# 绘制曲线的函数
def plot_losses(losses):
    epochs = losses['Epoch']
    for key, values in losses.items():
        if key != 'Epoch':
            plt.figure()
            plt.plot(epochs, values, label=key)
            plt.xlabel('Epoch')
            plt.ylabel(key)
            plt.title(f'{key} over Epochs')
            plt.legend()
            plt.grid(True)
            plt.show()


# 读取文本并绘制曲线
file_path = 'LossTrend/logs.log'  # 替换为你的文本文件路径
loss_data = parse_loss_data(file_path)
plot_losses(loss_data)

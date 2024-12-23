import argparse
import torch
import numpy as np
import random
from exp_classification import Exp_Classification


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # 创建 argument 解析器
    parser = argparse.ArgumentParser(description='PatchTST Classification')

    # 基本配置
    parser.add_argument('--task_name', type=str, default='classification', help='任务名称')
    parser.add_argument('--is_training', type=int, default=1, help='是否训练')
    parser.add_argument('--model_id', type=str, default='train', help='模型ID')
    parser.add_argument('--model', type=str, default='PatchTST', help='模型名称')

    # 数据加载配置
    parser.add_argument('--root_path', type=str, default='./', help='数据集根路径')
    parser.add_argument('--data_path', type=str, default='Heartbeat_TEST.ts', help='数据文件路径')

    # 数据集配置
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度')

    # 模型配置
    parser.add_argument('--enc_in', type=int, default=7, help='编码器输入特征维度')
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--e_layers', type=int, default=3, help='编码器层数')
    parser.add_argument('--d_ff', type=int, default=256, help='前馈网络的维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout率')

    # 优化配置
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--train_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--patience', type=int, default=10, help='早停轮数')

    # GPU配置
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU编号')

    args = parser.parse_args()
    Exp = Exp_Classification

    # 设置设备
    if args.use_gpu and torch.cuda.is_available():
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')

    print(f'Using device: {args.device}')

    # 开始训练或测试
    if args.is_training:
        for ii in range(1):  # 只进行一次训练
            exp = Exp(args)  # 创建Exp对象
            setting = f'{args.task_name}_{args.model_id}_{args.model}_train'
            print(f'Start training: {setting}')
            exp.train(setting)  # 训练模型

            print(f'Start testing: {setting}')
            exp.test(setting)  # 测试模型
    else:
        exp = Exp(args)  # 创建Exp对象
        setting = f'{args.task_name}_{args.model_id}_{args.model}_test'
        print(f'Start testing: {setting}')
        exp.test(setting, test=1)  # 测试模型

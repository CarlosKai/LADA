import torch
import torch.nn as nn
import numpy as np
import itertools    

from models.models import *
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, AUROC
from .TaskFusion import TaskFusion
from .GAT import GAT
from torch_geometric.data import Data, Batch


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class LALA(nn.Module):
    def __init__(self, backbone, configs, hparams, device):
        super(LALA, self).__init__()

        # 通用配置
        self.configs = configs
        self.hparams = hparams
        self.device = device

        # 构建模型
        self.la_taskFusion = TaskFusion(configs)
        self.la_tcn = TCN(configs)
        self.la_classifier = classifier(configs)
        # self.la_gat = GAT(configs)
        self.network = nn.Sequential(self.la_tcn, self.la_taskFusion, self.la_classifier)

        # 优化器和学习率调度器
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # Domain Discriminator 和其优化器
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        # 损失函数
        self.cross_entropy = nn.CrossEntropyLoss()

        # torchmetrics 指标
        self.ACC = Accuracy(task="multiclass", num_classes=configs.num_classes).to(self.device)
        self.F1 = F1Score(task="multiclass", num_classes=configs.num_classes, average="macro").to(self.device)
        self.AUROC = AUROC(task="multiclass", num_classes=configs.num_classes)

    def test_process(self, epoch, trg_test_loader, logger):
        trg_preds_list, trg_labels_list, trg_prob_list = [], [], []
        with torch.no_grad():
            self.la_tcn.eval()  # Set the model to eval mode
            self.la_classifier.eval()
            self.la_taskFusion.eval()
            self.domain_classifier.eval()

            for data, labels in trg_test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # Forward pass
                x1, _ = self.la_tcn(data)
                x2, _ = self.la_taskFusion(data)
                x2 = x2.mean(dim=1)
                predx = torch.cat([x1, x2], dim=1)
                predictions = self.la_classifier(predx)
                probabilities = torch.softmax(predictions, dim=1)
                pred = predictions.argmax(dim=1)  # Get the index of the max log-probability

                # Collect predictions and labels
                trg_preds_list.extend(pred.cpu().numpy())
                trg_labels_list.extend(labels.cpu().numpy())
                trg_prob_list.extend(probabilities.cpu().numpy())

        # Switch back to train mode
        self.la_tcn.train()
        self.la_classifier.train()
        self.la_taskFusion.train()
        self.domain_classifier.train()

        # Calculate Accuracy and F1-score
        trg_labels_list = torch.tensor(trg_labels_list).to(self.device)
        trg_preds_list = torch.tensor(trg_preds_list).to(self.device)
        trg_prob_list = torch.tensor(np.array(trg_prob_list)).to(self.device)
        trg_acc = self.ACC(trg_preds_list, trg_labels_list)
        trg_f1 = self.F1(trg_preds_list, trg_labels_list)
        trg_auc = self.AUROC(trg_prob_list, trg_labels_list)

        # Log the metrics
        logger.debug(f'[Epoch {epoch+1}/{self.hparams["num_epochs"]}] Target Domain Test Metrics:')
        logger.debug(f'Target Test Accuracy: {trg_acc:.4f}')
        logger.debug(f'Target Test F1-Score: {trg_f1:.4f}')

        return trg_acc, trg_f1, trg_auc


    def update(self, src_train_loader, src_test_loader, trg_train_loader, trg_test_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # training loop
            self.training_epoch(src_train_loader, src_test_loader, trg_train_loader, trg_test_loader, avg_meter, epoch)
            if (epoch + 1) % 10 == 0:
                _ = self.test_process(epoch, trg_test_loader, logger)
            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Trg_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model



    # 构造批处理图数据
    # def create_graph_data(self, tf_out, tcn_last, normalized_attns):
    #     data_list = []
    #     for i in range(tf_out.size(0)):  # 遍历 batch_size
    #         node_features = torch.cat([tf_out[i], tcn_last[i].unsqueeze(0)], dim=0)  # [num_nodes+1, feature_dim]
    #         num_nodes = node_features.size(0)
    #
    #         adj_matrix = normalized_attns[i]  # [num_nodes-1, num_nodes-1]
    #         # 扩展邻接矩阵，添加全局节点的边
    #         global_edges = torch.ones(num_nodes - 1, device=tf_out.device)  # 全局节点的边权重初始化为 1
    #         adj_matrix = torch.cat([adj_matrix, global_edges.unsqueeze(0)], dim=0)  # 添加到最后一行
    #         adj_matrix = torch.cat(
    #             [adj_matrix, torch.cat([global_edges, torch.tensor([0.], device=tf_out.device)]).unsqueeze(1)],
    #             dim=1)  # 添加到最后一列
    #
    #         # 构建边索引和边权重
    #         edge_index = torch.nonzero(adj_matrix > 0, as_tuple=False).t()  # [2, num_edges]
    #         edge_attr = adj_matrix[edge_index[0], edge_index[1]]  # [num_edges]
    #
    #         # 创建单个图
    #         graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    #         data_list.append(graph_data)

        # 合并为一个批次
        # return Batch.from_data_list(data_list)

    def training_epoch(self, src_train_loader, src_test_loader, trg_train_loader, trg_test_loader, avg_meter, epoch):
        if len(src_train_loader) > len(trg_train_loader):
            joint_loader =enumerate(zip(src_train_loader, itertools.cycle(trg_train_loader)))
        else:
            joint_loader =enumerate(zip(itertools.cycle(src_train_loader), trg_train_loader))

        num_batches = max(len(src_train_loader), len(trg_train_loader))

        for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loader:

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            src_x, src_y, trg_x, trg_y = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device), trg_y.to(self.device)
            src_tcn_last, src_tcn_feat = self.la_tcn(src_x)
            trg_tcn_last, trg_tcn_feat = self.la_tcn(trg_x)
            # src_tcn_pred = self.la_classifier(src_tcn_last)
            # trg_tcn_pred = self.la_classifier(trg_tcn_last)


            # TaskFusion and GAT
            src_tf_out, src_attns = self.la_taskFusion(src_x)
            # src_stacked_attns = torch.stack(src_attns, dim=1)
            # src_averaged_attns = src_stacked_attns.mean(dim=2).mean(dim=1)
            # src_mean = src_averaged_attns.mean(dim=(1, 2), keepdim=True)
            # src_std = src_averaged_attns.std(dim=(1, 2), keepdim=True)
            # src_normalized_attns = (src_averaged_attns - src_mean) / (src_std + 1e-5)

            trg_tf_out, trg_attns = self.la_taskFusion(trg_x)
            # trg_stacked_attns = torch.stack(trg_attns, dim=1)
            # trg_averaged_attns = trg_stacked_attns.mean(dim=2).mean(dim=1)
            # trg_mean = trg_averaged_attns.mean(dim=(1, 2), keepdim=True)
            # trg_std = trg_averaged_attns.std(dim=(1, 2), keepdim=True)
            # trg_normalized_attns = (trg_averaged_attns - trg_mean) / (trg_std + 1e-5)

            # 创建源域和目标域图数据
            # src_graph_data = self.create_graph_data(src_tf_out, src_tcn_last, src_normalized_attns)
            # trg_graph_data = self.create_graph_data(trg_tf_out, trg_tcn_last, trg_normalized_attns)

            # 运行 GAT 模型
            # src_output = self.la_gat(src_graph_data.x, src_graph_data.edge_index, src_graph_data.edge_attr)  # 源域输出
            # trg_output = self.la_gat(trg_graph_data.x, trg_graph_data.edge_index, trg_graph_data.edge_attr)  # 目标域输出
            #
            # # 输出最终结果
            # print("Source output shape:", src_output.shape)  # [num_nodes_total_in_batch, out_dim]
            # print("Target output shape:", trg_output.shape)  # [num_nodes_total_in_batch, out_dim]

            # src_tcn_last_expanded = src_tcn_last.unsqueeze(1)
            # src_combined = torch.cat([src_tf_out, src_tcn_last_expanded], dim=1)
            # src_flattened = src_combined.view(src_combined.size(0), -1)
            src_tf_out_merge = src_tf_out.mean(dim=1)  # [32, 128]
            src_combined = torch.cat([src_tf_out_merge, src_tcn_last], dim=1)

            # trg_tcn_last_expanded = trg_tcn_last.unsqueeze(1)
            # trg_combined = torch.cat([trg_tf_out, trg_tcn_last_expanded], dim=1)
            # trg_flattened = trg_combined.view(src_combined.size(0), -1)
            trg_tf_out_merge = trg_tf_out.mean(dim=1)  # [32, 128]
            trg_combined = torch.cat([trg_tf_out_merge, trg_tcn_last], dim=1)

            src_tcn_pred = self.la_classifier(src_combined)
            trg_tcn_pred = self.la_classifier(trg_combined)

            # src_cls_loss = self.cross_entropy(src_trans_logit, src_y)
            src_cls_loss = self.cross_entropy(src_tcn_pred, src_y)
            trg_cls_loss = self.cross_entropy(trg_tcn_pred, trg_y)

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_combined, alpha)
            # src_feat_reversed = ReverseLayerF.apply(src_tcn_last, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_combined, alpha)
            # trg_feat_reversed = ReverseLayerF.apply(trg_tcn_last, alpha)
            trg_domain_pred = self.domain_classifier(trg_feat_reversed)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                   self.hparams["domain_loss_wt"] * domain_loss

            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses = {'Src_cls_loss': src_cls_loss.item(), 'Trg_cls_loss':trg_cls_loss, 'Domain_loss': domain_loss.item(), 'Total_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()
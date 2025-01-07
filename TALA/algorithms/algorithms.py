import torch
import torch.nn as nn
import numpy as np
import itertools
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, AUROC
from .TaskFusion import TaskFusion
from .backbone import *
from .GNN import GNNTimeModel
from .Grad_TCN import GradTCN


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class TALA(nn.Module):
    def __init__(self, configs, hparams, device):
        super(TALA, self).__init__()
        # torch.autograd.set_detect_anomaly(True)
        # 通用配置
        self.configs = configs
        self.hparams = hparams
        self.device = device
        self.dataset_name = configs.dataset_name

        # 构建模型
        self.la_cnn = CNN(configs)
        self.la_taskFusion = TaskFusion(configs)
        self.la_tcn = TCN(configs)

        self.la_gnn = GNNTimeModel(configs)
        self.la_label_relation_classifier = Classifier1(configs)
        self.la_feature_classifier = Classifier2(configs)
        self.domain_classifier = Discriminator(configs)

        self.net1 = nn.Sequential(self.la_cnn)
        self.net2 = nn.Sequential(self.la_taskFusion, self.la_label_relation_classifier)
        self.net3 = nn.Sequential(self.la_gnn, self.la_tcn, self.la_feature_classifier)
        self.network = nn.Sequential(self.la_cnn, self.la_tcn, self.la_taskFusion, self.la_label_relation_classifier, self.la_gnn, self.la_feature_classifier, self.domain_classifier)
        # 优化器和学习率调度器
        self.optimizer1 = torch.optim.Adam(
            self.net1.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer2 = torch.optim.Adam(
            self.net2.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer3 = torch.optim.Adam(
            self.net3.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.lr_scheduler1 = StepLR(self.optimizer1, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler2 = StepLR(self.optimizer2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler3 = StepLR(self.optimizer3, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # 损失函数
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mmd_loss = MMD_loss()

        # torchmetrics 指标
        self.ACC = Accuracy(task="multiclass", num_classes=configs.num_classes).to(self.device)
        self.F1 = F1Score(task="multiclass", num_classes=configs.num_classes, average="macro").to(self.device)

    def test_process(self, loader_name, trg_test_loader, logger):
        trg_preds_list, trg_labels_list = [], []
        with torch.no_grad():
            self.la_cnn.eval()  # Set the model to eval mode
            self.la_taskFusion.eval()
            self.la_gnn.eval()
            self.la_tcn.eval()
            self.la_label_relation_classifier.eval()
            self.la_feature_classifier.eval()
            self.domain_classifier.eval()


            for data, labels in trg_test_loader:
                trg_x = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward
                n_vars = data.shape[1]
                n_batches = data.shape[0]
                trg_x = trg_x.reshape(n_batches * n_vars, 1, -1)
                trg_cnn_last = self.la_cnn(trg_x)
                trg_cnn_last = trg_cnn_last.reshape(n_batches, n_vars, -1)
                trg_gnn_feat = self.la_gnn(trg_cnn_last)
                trg_tcn_feat = self.la_tcn(trg_gnn_feat.permute(0,2,1))
                predictions = self.la_feature_classifier(trg_tcn_feat)
                pred = predictions.argmax(dim=1)

                # Collect predictions and labels
                trg_preds_list.extend(pred.cpu().numpy())
                trg_labels_list.extend(labels.cpu().numpy())


        # Switch back to train mode
        self.la_cnn.train()
        self.la_taskFusion.train()
        self.la_gnn.train()
        self.la_tcn.train()
        self.la_label_relation_classifier.train()
        self.la_feature_classifier.train()
        self.domain_classifier.train()

        # Calculate Accuracy and F1-score
        trg_labels_list = torch.tensor(trg_labels_list).to(self.device)
        trg_preds_list = torch.tensor(trg_preds_list).to(self.device)

        trg_acc = self.ACC(trg_preds_list, trg_labels_list)
        trg_f1 = self.F1(trg_preds_list, trg_labels_list)


        # Log the metrics
        logger.debug(f'{loader_name} Accuracy: {trg_acc:.4f}')
        logger.debug(f'{loader_name} F1-Score: {trg_acc:.4f}')


        return trg_acc, trg_f1


    def update(self, src_train_loader, src_test_loader, trg_train_loader, trg_test_loader, avg_meter, logger):
        # defining best and last model
        best_trg_risk = torch.tensor(0.0, device=self.device)
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # training loop
            trg_preds_list, trg_labels_list = [], []
            self.training_epoch(src_train_loader, src_test_loader, trg_train_loader, trg_test_loader, avg_meter, epoch)

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            if (epoch + 1) % 5 == 0:
                _ = self.test_process("Source Domain Train", src_train_loader, logger)
                _ = self.test_process("Target Domain Train", trg_train_loader, logger)
                tmp_trg_risk, _ = self.test_process("Target Domain Test", trg_test_loader, logger)
                if tmp_trg_risk > best_trg_risk:
                    best_trg_risk = tmp_trg_risk
                    best_model = deepcopy(self.network.state_dict())

            # print log
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model



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
            n_vars = src_x.shape[1]
            n_batches = src_x.shape[0]

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_x = src_x.reshape(n_batches * n_vars, 1, -1)
            trg_x = trg_x.reshape(n_batches * n_vars, 1, -1)
            src_cnn_last = self.la_cnn(src_x)
            trg_cnn_last = self.la_cnn(trg_x)
            src_cnn_last = src_cnn_last.reshape(n_batches, n_vars, -1)
            trg_cnn_last = trg_cnn_last.reshape(n_batches, n_vars, -1)

            _, src_fusion_attn = self.la_taskFusion(src_cnn_last)
            _, trg_fusion_attn = self.la_taskFusion(trg_cnn_last)
            src_fusion_attn = src_fusion_attn.reshape(n_batches, -1)
            trg_fusion_attn = trg_fusion_attn.reshape(n_batches, -1)

            src_gnn_feat = self.la_gnn(src_cnn_last)
            trg_gnn_feat = self.la_gnn(trg_cnn_last)

            src_gnn_feat = src_gnn_feat.permute(0, 2, 1)
            trg_gnn_feat = trg_gnn_feat.permute(0, 2, 1)


            src_task_pred = self.la_label_relation_classifier(src_fusion_attn)
            trg_task_pred = self.la_label_relation_classifier(trg_fusion_attn)

            src_task_loss = self.cross_entropy(src_task_pred, src_y)

            # trg_task_loss = self.cross_entropy(trg_task_pred, trg_y)

            src_tcn_feat = self.la_tcn(src_gnn_feat)
            trg_tcn_feat = self.la_tcn(trg_gnn_feat)

            src_final_pred = self.la_feature_classifier(src_tcn_feat)
            trg_final_pred = self.la_feature_classifier(trg_tcn_feat)

            # Grad_TCN
            src_masked_irrelated_tcn_pred, src_masked_related_tcn_pred  = self.count_batch_Grad_TCN(src_gnn_feat, src_y)

            src_label_max = src_final_pred.argmax(dim=1)
            src_label_min = src_final_pred.argmin(dim=1)
            trg_label_max = trg_final_pred.argmax(dim=1)
            trg_label_min = trg_final_pred.argmin(dim=1)

            src_f_pos = self.get_cls_weiight(src_tcn_feat, src_y)
            src_f_pos_label_max = self.get_cls_weiight(src_tcn_feat, src_label_max)
            trg_f_pos = self.get_cls_weiight(trg_tcn_feat, trg_label_max)
            src_f_neg = self.get_cls_weiight(src_tcn_feat, src_label_min)
            trg_f_neg = self.get_cls_weiight(trg_tcn_feat, trg_label_min)

            loss_indi = self.instance_contrastive_loss_v3(src_final_pred, src_masked_irrelated_tcn_pred, src_masked_related_tcn_pred) /2.0
            # loss_indi = torch.tensor(0).to(self.device)
            loss_inst = self.instance_contrastive_loss_v3(src_tcn_feat, src_f_pos, [trg_f_pos, trg_f_neg])
            # contrast_loss = loss_inst + loss_indi
            contrast_loss =  loss_inst * self.hparams["contrast_inst_loss_wt"]  + loss_indi * self.hparams["contrast_indi_loss_wt"]

            src_cls_loss = self.cross_entropy(src_final_pred, src_y)
            trg_cls_loss = self.cross_entropy(trg_final_pred, trg_y)

            # Domain classification loss
            # source
            # src_feat_reversed = ReverseLayerF.apply(src_tcn_feat, alpha)
            src_feat_reversed = ReverseLayerF.apply(src_f_pos, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_tcn_feat, alpha)
            trg_domain_pred = self.domain_classifier(trg_feat_reversed)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            src_task_loss = self.hparams["src_task_loss_wt"] * src_task_loss
            trg_mmd_loss = self.hparams["trg_mmd_loss_wt"] * self.mmd_loss(trg_task_pred, trg_final_pred)

            # 不加入mmd损失的情况
            # loss1 = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
            #        self.hparams["domain_loss_wt"] * domain_loss
            #
            # loss2 = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
            #        self.hparams["domain_loss_wt"] * domain_loss  +  src_task_loss
            #
            # self.optimizer2.zero_grad()
            # self.optimizer1.zero_grad()
            # src_task_loss.backward(retain_graph=True)
            # self.optimizer2.step()
            # self.optimizer3.zero_grad()
            # self.optimizer_disc.zero_grad()
            # loss1.backward()
            # self.optimizer3.step()
            # self.optimizer_disc.step()
            # self.optimizer1.step()
            # losses = {'Src_cls_loss': src_cls_loss.item(), 'Trg_cls_loss': trg_cls_loss,
            #           'Domain_loss': domain_loss.item(),
            #           'Src_task_loss': src_task_loss.item(), 'class_feature_and_domain_loss': loss1.item(),
            #           'Total_loss': loss2.item()}

            if self.configs.dataset_name == "EEG":
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                self.optimizer3.zero_grad()
                loss2 = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                        self.hparams["domain_loss_wt"] * domain_loss + self.hparams[
                            "dual_contrastive_loss_wt"] * contrast_loss
                loss2.backward()
                self.optimizer3.step()
                self.optimizer_disc.step()
                self.optimizer1.step()
                self.optimizer2.step()
                losses = {'Src_cls_loss': src_cls_loss.item(), 'Trg_cls_loss': trg_cls_loss.item(),
                          'Domain_loss': domain_loss.item(),
                          'Contrast_loss': contrast_loss.item(), 'Contrast_loss_inst': loss_inst.item(),
                          'Contrast_loss_indi': loss_indi.item(), }
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)
                continue

            #加入目标域mmd损失的情况
            loss1 = src_task_loss + trg_mmd_loss
            loss2 = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                    self.hparams["domain_loss_wt"] * domain_loss + self.hparams["dual_contrastive_loss_wt"] * contrast_loss

            loss3 = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                    self.hparams["domain_loss_wt"] * domain_loss + src_task_loss + trg_mmd_loss
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss1.backward(retain_graph=True)
            self.optimizer2.step()
            self.optimizer3.zero_grad()
            self.optimizer_disc.zero_grad()
            loss2.backward()
            self.optimizer3.step()
            self.optimizer_disc.step()
            self.optimizer1.step()

            losses = {'Src_cls_loss': src_cls_loss.item(), 'Trg_cls_loss':trg_cls_loss.item(),
                      'Domain_loss': domain_loss.item(), 'Src_task_loss': src_task_loss.item(),
                      'class_feature_and_domain_loss': loss1.item(), 'Trg_MMD_loss': trg_mmd_loss,
                      'Contrast_loss': contrast_loss.item(), 'Contrast_loss_inst': loss_inst.item(),
                      'Contrast_loss_indi': loss_indi.item(), 'Total_loss': loss3.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler1.step()
        self.lr_scheduler2.step()
        self.lr_scheduler3.step()

    def get_cls_weiight(self, f, labels):
        # 计算src_feat权重
        w = self.la_feature_classifier.logits.weight[labels].detach()
        eng_before = (f ** 2).sum(dim=1, keepdim=True)  # 原始能量 [B, 1]
        eng_after = ((f * w) ** 2).sum(dim=1, keepdim=True)  # 调整后的能量 [B, 1]
        scalar = (eng_before / eng_after).sqrt()  # 计算缩放因子
        w_pos = w * scalar  # 对齐后的权重
        f_pos = f * w_pos
        return f_pos


    def instance_contrastive_loss_v3(self, z1, z2, negatives):
        """
        z1: 主特征 (正样本对的一部分)，shape: (batch_size, time_steps)
        z2: 正样本对，shape: (batch_size, time_steps)
        negatives: 负样本对列表 [z3, z4, ...]，每个 shape: (batch_size, time_steps)
        """
        # 计算正样本对相似度
        sim_pos = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(1), dim=1)  # shape: (batch_size, time_steps)
        # 计算负样本对相似度
        sim_neg_list = [F.cosine_similarity(z1.unsqueeze(1), neg.unsqueeze(1), dim=1) for neg in negatives]
        sim_neg = torch.stack(sim_neg_list, dim=0)  # shape: (num_negatives, batch_size, time_steps)
        # 计算分母：正样本和所有负样本相似度的指数和
        denominator = torch.exp(sim_pos) + torch.sum(torch.exp(sim_neg), dim=0)  # shape: (batch_size, time_steps)
        # 计算对比损失
        loss = -torch.log(torch.exp(sim_pos) / denominator)  # shape: (batch_size, time_steps)
        loss = loss.mean()  # 对时间步和 batch 平均

        return loss

    def count_batch_Grad_TCN(self, src_gnn_feat, src_y):
        self.la_tcn.eval()
        self.la_feature_classifier.eval()
        target_layer = self.la_tcn.conv_block2[0]  # Second convolution layer
        grad_cam = GradTCN(self.la_tcn, self.la_feature_classifier, target_layer)

        # Generate CAM and mask important segments
        time_step_importances = grad_cam.generate_cam(src_gnn_feat, src_y)
        masked_irrelated_inputs, masked_related_inputs = grad_cam.mask_important_segments(src_gnn_feat, time_step_importances)

        # Pass masked inputs through the model
        masked_related_outputs = self.la_tcn(masked_related_inputs)
        masked_irrelated_outputs = self.la_tcn(masked_irrelated_inputs)
        masked_related_outputs = self.la_feature_classifier(masked_related_outputs)
        masked_irrelated_outputs = self.la_feature_classifier(masked_irrelated_outputs)
        self.la_tcn.train()
        self.la_feature_classifier.train()
        self.optimizer3.zero_grad()
        return masked_irrelated_outputs, masked_related_outputs



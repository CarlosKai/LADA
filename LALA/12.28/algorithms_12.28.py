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
from algorithms.TaskFusion import TaskFusion
from algorithms.GAT import GAT
from torch_geometric.data import Data, Batch
from torchsummary import summary


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
        self.la_cnn = CNN(configs)
        self.la_classifier = classifier(configs)
        # self.la_gat = GAT(configs)

        self.net1 = nn.Sequential(self.la_tcn, self.la_taskFusion, self.la_classifier, self.la_cnn)
        # 优化器和学习率调度器
        self.optimizer = torch.optim.Adam(
            self.net1.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        # Domain Discriminator 和其优化器
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.lr_scheduler1 = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler2 = StepLR(self.optimizer_disc, step_size=hparams['step_size'], gamma=hparams['lr_decay'])



        self.network = nn.Sequential(self.la_tcn, self.la_taskFusion, self.la_classifier, self.domain_classifier)

        # 损失函数
        self.cross_entropy = nn.CrossEntropyLoss()

        # torchmetrics 指标
        self.ACC = Accuracy(task="multiclass", num_classes=configs.num_classes).to(self.device)
        self.F1 = F1Score(task="multiclass", num_classes=configs.num_classes, average="macro").to(self.device)
        self.AUROC = AUROC(task="multiclass", num_classes=configs.num_classes)

    def test_process(self, loader_name, trg_test_loader, logger):
        trg_preds_list, trg_labels_list, trg_prob_list = [], [], []
        with torch.no_grad():
            self.la_tcn.eval()  # Set the model to eval mode
            self.la_classifier.eval()
            self.la_taskFusion.eval()
            self.domain_classifier.eval()
            self.la_cnn.eval()

            for data, labels in trg_test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # print(summary(self.la_tcn, (9,128)))
                # print(summary(self.la_taskFusion, (9,128)))

                # Forward pass training_epoch_with_taskfusiion
                # x2, _ = self.la_taskFusion(data)
                # x2 = x2.reshape(x2.shape[0], -1)
                # predictions = self.la_classifier(x2)
                # probabilities = torch.softmax(predictions, dim=1)
                # pred = predictions.argmax(dim=1)

                # Forward pass training_epoch_with_tcn
                # x1, _ = self.la_tcn(data)
                # predictions = self.la_classifier(x1)
                # probabilities = torch.softmax(predictions, dim=1)
                # pred = predictions.argmax(dim=1)

                # Forward pass training_epoch_with_taskfusiion+cnn
                # x2, _ = self.la_taskFusion(data)
                # x2, _ = self.la_cnn(x2)
                # predictions = self.la_classifier(x2)
                # probabilities = torch.softmax(predictions, dim=1)
                # pred = predictions.argmax(dim=1)

                # Forward pass training_epoch_with_taskfusiion_concat_cnn
                x1, x1_attns = self.la_taskFusion(data)
                x2, _ = self.la_cnn(x1)
                x3, _ = self.la_cnn(x1_attns)
                predictions = self.la_classifier(torch.cat((x2, x3), dim=1))
                probabilities = torch.softmax(predictions, dim=1)
                pred = predictions.argmax(dim=1)

                # Collect predictions and labels
                trg_preds_list.extend(pred.cpu().numpy())
                trg_labels_list.extend(labels.cpu().numpy())
                trg_prob_list.extend(probabilities.cpu().numpy())

        # Switch back to train mode
        self.la_tcn.train()
        self.la_classifier.train()
        self.la_taskFusion.train()
        self.domain_classifier.train()
        self.la_cnn.train()

        # Calculate Accuracy and F1-score
        trg_labels_list = torch.tensor(trg_labels_list).to(self.device)
        trg_preds_list = torch.tensor(trg_preds_list).to(self.device)
        trg_prob_list = torch.tensor(np.array(trg_prob_list)).to(self.device)
        trg_acc = self.ACC(trg_preds_list, trg_labels_list)
        trg_f1 = self.F1(trg_preds_list, trg_labels_list)
        trg_auc = self.AUROC(trg_prob_list, trg_labels_list)

        # Log the metrics
        logger.debug(f'{loader_name} Accuracy: {trg_acc:.4f}')
        logger.debug(f'{loader_name} F1-Score: {trg_acc:.4f}')


        return trg_acc, trg_f1, trg_auc


    def update(self, src_train_loader, src_test_loader, trg_train_loader, trg_test_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # training loop
            # self.training_epoch_with_tcn(src_train_loader, src_test_loader, trg_train_loader, trg_test_loader,avg_meter, epoch)
            self.training_epoch_with_taskfusion_concat_tocnn(src_train_loader, src_test_loader, trg_train_loader, trg_test_loader, avg_meter, epoch)
            # self.training_epoch_with_taskfusion(src_train_loader, src_test_loader, trg_train_loader, trg_test_loader,avg_meter, epoch)
            # self.training_epoch(src_train_loader, src_test_loader, trg_train_loader, trg_test_loader, avg_meter, epoch)

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            if (epoch + 1) % 5 == 0:
                _ = self.test_process("Source Domain Train", src_train_loader, logger)
                _ = self.test_process("Target Domain Train", trg_train_loader, logger)
                _ = self.test_process("Target Domain Test", trg_test_loader, logger)
            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Trg_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())


            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model


    def training_epoch_with_taskfusion_concat_tocnn(self, src_train_loader, src_test_loader, trg_train_loader, trg_test_loader, avg_meter, epoch):
        if len(src_train_loader) > len(trg_train_loader):
            joint_loader =enumerate(zip(src_train_loader, itertools.cycle(trg_train_loader)))
        else:
            joint_loader =enumerate(zip(itertools.cycle(src_train_loader), trg_train_loader))

        num_batches = max(len(src_train_loader), len(trg_train_loader))

        for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loader:

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            src_x, src_y, trg_x, trg_y = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device), trg_y.to(self.device)

            # TaskFusion
            src_tf_out, src_attns = self.la_taskFusion(src_x)
            trg_tf_out, trg_attns = self.la_taskFusion(trg_x)
            # src_x_concat = torch.cat((src_x, src_attns), dim=2)
            # trg_x_concat = torch.cat((trg_x, trg_attns), dim=2)

            # src_x_concat = torch.cat((src_x, src_x + src_x.mul(src_attns)), dim=2)
            # trg_x_concat = torch.cat((trg_x, trg_x + trg_x.mul(trg_attns)), dim=2)

            src_cnn_out, _ = self.la_cnn(src_x)
            trg_cnn_out, _ = self.la_cnn(trg_x)
            src_cnn_attn_out, _ = self.la_cnn(src_attns)
            trg_cnn_attn_out, _ = self.la_cnn(trg_attns)
            # x2, _ = self.la_cnn(x1)
            # x3, _ = self.la_cnn(x1_attns)
            src_cnn_out = torch.cat((src_cnn_out, src_cnn_attn_out), dim=1)
            trg_cnn_out = torch.cat((trg_cnn_out, trg_cnn_attn_out), dim=1)


            # src_tf_out_merge = src_tf_out.reshape(src_tf_out.shape[0], -1)
            # trg_tf_out_merge = trg_tf_out.reshape(trg_tf_out.shape[0], -1)

            src_final_pred = self.la_classifier(src_cnn_out)
            trg_final_pred = self.la_classifier(trg_cnn_out)

            src_cls_loss = self.cross_entropy(src_final_pred, src_y)
            trg_cls_loss = self.cross_entropy(trg_final_pred, trg_y)

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_cnn_out, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_cnn_out, alpha)
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




    # 只使用taskfusion网络输入cnn进行训练
    def training_epoch_with_taskfusion(self, src_train_loader, src_test_loader, trg_train_loader, trg_test_loader, avg_meter, epoch):
        if len(src_train_loader) > len(trg_train_loader):
            joint_loader =enumerate(zip(src_train_loader, itertools.cycle(trg_train_loader)))
        else:
            joint_loader =enumerate(zip(itertools.cycle(src_train_loader), trg_train_loader))

        num_batches = max(len(src_train_loader), len(trg_train_loader))

        for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loader:

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            src_x, src_y, trg_x, trg_y = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device), trg_y.to(self.device)

            # TaskFusion
            src_tf_out, src_attns = self.la_taskFusion(src_x)
            trg_tf_out, trg_attns = self.la_taskFusion(trg_x)

            src_cnn_out, _ = self.la_cnn(src_tf_out)
            trg_cnn_out, _ = self.la_cnn(trg_tf_out)


            # src_tf_out_merge = src_tf_out.reshape(src_tf_out.shape[0], -1)
            # trg_tf_out_merge = trg_tf_out.reshape(trg_tf_out.shape[0], -1)

            src_final_pred = self.la_classifier(src_cnn_out)
            trg_final_pred = self.la_classifier(trg_cnn_out)

            src_cls_loss = self.cross_entropy(src_final_pred, src_y)
            trg_cls_loss = self.cross_entropy(trg_final_pred, trg_y)

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_cnn_out, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_cnn_out, alpha)
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


    def training_epoch_with_tcn(self, src_train_loader, src_test_loader, trg_train_loader, trg_test_loader, avg_meter, epoch):
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
            src_final_pred = self.la_classifier(src_tcn_last)
            trg_final_pred = self.la_classifier(trg_tcn_last)

            src_cls_loss = self.cross_entropy(src_final_pred, src_y)
            trg_cls_loss = self.cross_entropy(trg_final_pred, trg_y)

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            # Domain classification loss
            src_feat_reversed = ReverseLayerF.apply(src_tcn_last, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_tcn_last, alpha)
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

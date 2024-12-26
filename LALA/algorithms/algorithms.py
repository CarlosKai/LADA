import torch
import torch.nn as nn
import numpy as np
import itertools    

from models.models import *
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from .PatchTST import PatchTST
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torchmetrics import Accuracy, AUROC, F1Score


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
        self.la_patchTST = PatchTST(configs)
        self.la_tcn = backbone(configs)
        self.la_classifier = classifier(configs)
        self.network = nn.Sequential(self.la_tcn, self.la_patchTST, self.la_classifier)

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

    def update(self, src_train_loader, src_test_loader, trg_train_loader, trg_test_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # training loop
            self.training_epoch(src_train_loader, src_test_loader, trg_train_loader, trg_test_loader, avg_meter, epoch)

            # Evaluate on trg_test_loader every 10 epochs
            if (epoch + 1) % 10 == 0:
                trg_preds_list, trg_labels_list = [], []
                with torch.no_grad():
                    self.la_tcn.eval()  # Set the model to eval mode
                    self.la_classifier.eval()
                    for data, labels in trg_test_loader:
                        data = data.float().to(self.device)
                        labels = labels.view((-1)).long().to(self.device)

                        # Forward pass
                        data, _ = self.la_tcn(data)
                        predictions = self.la_classifier(data)
                        pred = predictions.argmax(dim=1)  # Get the index of the max log-probability

                        # Collect predictions and labels
                        trg_preds_list.extend(pred.cpu().numpy())
                        trg_labels_list.extend(labels.cpu().numpy())
                # Switch back to train mode
                self.la_tcn.train()
                self.la_classifier.train()
                # Calculate Accuracy and F1-score
                trg_labels_list = torch.tensor(trg_labels_list).to(self.device)
                trg_preds_list = torch.tensor(trg_preds_list).to(self.device)
                trg_acc = self.ACC(trg_preds_list, trg_labels_list)
                trg_f1 = self.F1(trg_preds_list, trg_labels_list)

                # Log the metrics
                logger.debug(f'[Epoch {epoch+1}/{self.hparams["num_epochs"]}] Target Domain Test Metrics:')
                logger.debug(f'Target Test Accuracy: {trg_acc:.4f}')
                logger.debug(f'Target Test F1-Score: {trg_f1:.4f}')

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
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
            src_tcn_last, src_tcn_feat = self.la_tcn(src_x)
            trg_tcn_last, trg_tcn_feat = self.la_tcn(trg_x)
            src_pred = self.la_classifier(src_tcn_last)
            trg_pred = self.la_classifier(trg_tcn_last)

            # src_trans_logit, src_trans_feat = self.la_patchTST(src_tcn_feat)
            # trg_trans_logit, trg_trans_feat = self.la_patchTST(src_tcn_feat)

            # src_cls_loss = self.cross_entropy(src_trans_logit, src_y)
            src_cls_loss = self.cross_entropy(src_pred, src_y)
            trg_cls_loss = self.cross_entropy(trg_pred, trg_y)

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            # Domain classification loss
            # source
            # src_feat_reversed = ReverseLayerF.apply(src_trans_feat, alpha)
            src_feat_reversed = ReverseLayerF.apply(src_tcn_last, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            # trg_feat_reversed = ReverseLayerF.apply(trg_trans_feat, alpha)
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


from algorithms.algorithms import *
from algorithms.PatchTST import PatchTST

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class TLA(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.feature_extractor = PatchTST(configs, patch_len=16, stride=32)
        self.network = nn.Sequential(self.feature_extractor)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        for src_x, src_y in src_loader:

            src_x, src_y = src_x.to(self.device), src_y.to(self.device)
            src_feat = self.feature_extractor(src_x)
            src_pred = src_feat

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


    # def __init__(self, backbone, configs, hparams, device):
    #     super(TLA, self).__init__(configs, backbone)
    #
    #     self.d_model = configs.d_model
    #     self.n_heads = configs.n_heads
    #     self.num_layers = configs.num_layers
    #     self.num_classes = configs.num_classes
    #
    #     # 通过 `backbone` 获取两个 Transformer 模型
    #     self.src_transformer = nn.TransformerEncoder(
    #         encoder_layer=nn.TransformerEncoderLayer(
    #             d_model=self.d_model,
    #             nhead=self.n_heads,
    #             batch_first=True,
    #             # dim_feedforward=self.d_ff,  # 如果需要，你可以指定全连接层的维度
    #             # dropout=self.dropout
    #         ),
    #         num_layers=self.num_layers
    #     )
    #
    #     self.trg_transformer = nn.TransformerEncoder(
    #         encoder_layer=nn.TransformerEncoderLayer(
    #             d_model=self.d_model,
    #             nhead=self.n_heads,
    #             batch_first=True,
    #             # dim_feedforward=self.d_ff,  # 如果需要，你可以指定全连接层的维度
    #             # dropout=self.dropout
    #         ),
    #         num_layers=self.num_layers
    #     )
    #
    #     self.hparams = hparams
    #
    #     # 分类头（用于源域分类）
    #     self.classifier = classifier(configs)
    #
    #     # 领域判别器（用于对抗学习）
    #     self.domain_classifier = Discriminator(configs)
    #
    #     # 优化器
    #     self.optimizer_fe = torch.optim.Adam(
    #         list(self.src_transformer.parameters()) + list(self.trg_transformer.parameters()) + list(
    #             self.classifier.parameters()),
    #         lr=hparams["learning_rate"],
    #         weight_decay=hparams["weight_decay"]
    #     )
    #     self.optimizer_disc = torch.optim.Adam(
    #         self.domain_classifier.parameters(),
    #         lr=hparams["learning_rate"],
    #         weight_decay=hparams["weight_decay"]
    #     )
    #
    #     # 学习率调度器
    #     self.lr_scheduler_fe = StepLR(self.optimizer_fe, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
    #     self.lr_scheduler_disc = StepLR(self.optimizer_disc, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
    #
    #     # 损失函数
    #     self.cross_entropy = nn.CrossEntropyLoss()
    #     self.mmd_loss = MMD_loss()  # 或者使用其他对抗学习损失函数
    #
    #     self.device = device
    #
    # def update(self, src_loader, trg_loader, avg_meter, logger):
    #     best_model = None
    #     best_src_risk = float('inf')
    #
    #     for epoch in range(1, self.hparams["num_epochs"] + 1):
    #         # 训练循环
    #         self.training_epoch(src_loader, trg_loader, avg_meter, epoch)
    #
    #         # 保存最佳模型
    #         if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
    #             best_src_risk = avg_meter['Src_cls_loss'].avg
    #             best_model = deepcopy(self.network.state_dict())
    #
    #         # 日志输出
    #         logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
    #         for key, val in avg_meter.items():
    #             logger.debug(f'{key}\t: {val.avg:2.4f}')
    #         logger.debug(f'-------------------------------------')
    #
    #     last_model = self.network.state_dict()
    #
    #     return last_model, best_model
    #
    # def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
    #     joint_loader = enumerate(zip(src_loader, trg_loader))
    #
    #     for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
    #         src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
    #
    #         # 处理源域数据
    #         src_feat = self.src_transformer(src_x)
    #         src_pred = self.classifier(src_feat)  # 源域分类预测
    #
    #         # 计算源域的分类损失
    #         # 聚合时间步输出，取每个时间步的最大值
    #         src_pred_aggregated = src_pred.max(dim=1)[0]  # 形状变为 [batch_size, num_classes]
    #
    #         # 计算分类损失
    #         src_cls_loss = self.cross_entropy(src_pred_aggregated, src_y)
    #         # src_cls_loss = self.cross_entropy(src_pred, src_y)
    #
    #         # 处理目标域数据
    #         trg_feat = self.trg_transformer(trg_x)
    #
    #         # 对抗损失
    #         src_domain_label = torch.ones(len(src_x)).to(self.device)
    #         trg_domain_label = torch.zeros(len(trg_x)).to(self.device)
    #
    #         p = float(step + epoch * len(src_loader)) / (self.hparams["num_epochs"] * len(src_loader))
    #         alpha = 2. / (1. + np.exp(-10 * p)) - 1  # 常用的 alpha 计算公式
    #
    #         src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
    #         src_domain_pred = self.domain_classifier(src_feat_reversed)
    #         src_domain_pred_aggregated = src_pred.max(dim=1)[0]  # 形状变为 [batch_size, num_classes]
    #         src_domain_loss = self.cross_entropy(src_domain_pred_aggregated, src_domain_label.long())
    #
    #         trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
    #         trg_domain_pred = self.domain_classifier(trg_feat_reversed)
    #         trg_domain_red_aggregated = src_pred.max(dim=1)[0]  # 形状变为 [batch_size, num_classes]
    #         trg_domain_loss = self.cross_entropy(trg_domain_red_aggregated, trg_domain_label.long())
    #
    #         # 总对抗损失
    #         domain_loss = src_domain_loss + trg_domain_loss
    #
    #         # 计算总损失
    #         loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss
    #         # loss =  src_cls_loss* +  domain_loss
    #
    #         # 反向传播和优化
    #         self.optimizer_fe.zero_grad()
    #         self.optimizer_disc.zero_grad()
    #
    #         loss.backward()
    #
    #         self.optimizer_fe.step()
    #         self.optimizer_disc.step()
    #
    #         # 更新损失
    #         losses = {'Total_loss': loss.item(), 'Src_cls_loss': src_cls_loss.item(), 'Domain_loss': domain_loss.item()}
    #         for key, val in losses.items():
    #             avg_meter[key].update(val, 32)
    #
    #     # 学习率更新
    #     self.lr_scheduler_fe.step()
    #     self.lr_scheduler_disc.step()
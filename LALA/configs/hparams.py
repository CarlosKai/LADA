## The cuurent hyper-parameters values are not necessarily the best ones for a specific risk.
def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 2,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5

        }
        self.alg_hparams = {
            "LALA": {
                # "domain_loss_wt": 2.943729820531079,
                "domain_loss_wt": 1,
                "learning_rate": 0.001,
                "src_task_loss_wt": 1,
                "trg_mmd_loss_wt": 1,
                # "src_cls_loss_wt": 5.1390077646202,
                "src_cls_loss_wt": 2,
                'dual_contrastive_loss_wt': 1,
                "contrast_indi_loss_wt" : 0.5,
                "contrast_inst_loss_wt": 0.5,
                "weight_decay": 0.0001
            }
        }
        self.ablation_params = {
            'label_distribution_align': True,
            'task_relation_classify': True,
            'label_attentive_align': True,
            'dual_contrastive_inst_loss': True,
            'dual_contrastive_indi_grad_cam_loss': True
        }


class EEG():
    def __init__(self):
        super().__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 128,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5

        }
        self.alg_hparams = {
            "LALA": {
                "domain_loss_wt": 1,
                "learning_rate": 0.001,
                "src_task_loss_wt": 1,
                "trg_mmd_loss_wt": 1,
                "src_cls_loss_wt": 3,
                'dual_contrastive_loss_wt': 1,
                "contrast_indi_loss_wt": 0.5,
                "contrast_inst_loss_wt": 0.5,
                "weight_decay": 0.0001
            }
        }
        self.ablation_params = {
            'label_distribution_align': True,
            'task_relation_classify': True,
            'label_attentive_align': True,
            'dual_contrastive_inst_loss': True,
            'dual_contrastive_indi_grad_cam_loss': True
        }


class WISDM():
    def __init__(self):
        super().__init__()
        self.train_params = {
            'num_epochs': 200,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5

        }
        self.alg_hparams = {
            "LALA": {
                "domain_loss_wt": 1,
                "learning_rate": 0.001,
                "src_task_loss_wt": 1,
                "trg_mmd_loss_wt": 1,
                "src_cls_loss_wt": 2,
                'dual_contrastive_loss_wt': 1,
                "contrast_indi_loss_wt": 0.5,
                "contrast_inst_loss_wt": 0.5,
                "weight_decay": 0.0001
            }
        }
        self.ablation_params = {
            'label_distribution_align': True,
            'task_relation_classify': True,
            'label_attentive_align': True,
            'dual_contrastive_inst_loss': True,
            'dual_contrastive_indi_grad_cam_loss': True
        }


class HHAR_SA():
    def __init__(self):
        super().__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5

        }
        self.alg_hparams = {
            "LALA": {
                "domain_loss_wt": 1,
                "learning_rate": 0.001,
                "src_task_loss_wt": 1,
                "trg_mmd_loss_wt": 1,
                "src_cls_loss_wt": 3,
                'dual_contrastive_loss_wt': 1,
                "contrast_indi_loss_wt" : 0.5,
                "contrast_inst_loss_wt": 0.5,
                "weight_decay": 0.0001
            }
        }
        self.ablation_params = {
            'label_distribution_align': True,
            'task_relation_classify': True,
            'label_attentive_align': True,
            'dual_contrastive_inst_loss': True,
            'dual_contrastive_indi_grad_cam_loss': True
        }

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
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5

        }
        self.alg_hparams = {
            "LALA": {
                "domain_loss_wt": 2.943729820531079,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 5.1390077646202,
                "weight_decay": 0.0001
            }
        }
        self.ablation_params = {
            'grad_tcn': True,
            'relation_pool': True,
            'super_node': True
        }


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 128,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5

        }
        self.alg_hparams = {
            "LALA": {
                "domain_loss_wt": 2.943729820531079,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 5.1390077646202,
                "weight_decay": 0.0001
            }
        }
        self.ablation_params = {
            'grad_tcn': True,
            'relation_pool': True,
            'super_node': True
        }


class WISDM():
    def __init__(self):
        super().__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,

        }
        self.alg_hparams = {
            "LALA": {
                "domain_loss_wt": 2.943729820531079,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 5.1390077646202,
                "weight_decay": 0.0001
            }
        }
        self.ablation_params = {
            'grad_tcn': True,
            'relation_pool': True,
            'super_node': True
        }


class HHAR():
    def __init__(self):
        super().__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
        }
        self.alg_hparams = {
            "LALA": {
                "domain_loss_wt": 2.943729820531079,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 5.1390077646202,
                "weight_decay": 0.0001
            }
        }
        self.ablation_params = {
            'grad_tcn': True,
            'relation_pool': True,
            'super_node': True
        }

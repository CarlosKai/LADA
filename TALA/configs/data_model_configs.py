def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class HAR():
    def __init__(self):
        super().__init__()
        self.dataset_name = 'HAR'
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']

        # self.scenarios = []
        # for i in range(1, 20):
        #     for j in range(1, 20):
        #         if i != j:
        #             self.scenarios.append((str(i), str(j)))

        self.scenarios = [("19", "25")]
        # self.scenarios = [
        #     ("15", "19"),
        #     ("18", "21"),
        #     ("19", "25"),
        #     ("19", "27"),
        #     ("20", "6"),
        #     ("23", "13"),
        #     ("24", "22"),
        #     ("25", "24"),
        #     ("3", "20"),
        #     ("13", "19"),
        # ]


        # self.scenarios = [("2", "11"), ("6", "23"), ("7", "13"), ("9", "18"), ("12", "16"),
        #                   ("13", "19"), ("18", "21"), ("20", "6"), ("23", "13"), ("24", "12")]

        # base information
        self.sequence_len = 128
        self.num_classes = 6
        self.input_channels = 9
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # task_fusion
        self.d_model = 128  # 输入transformer的特征维度
        self.n_heads = 3  # 有几个head
        self.e_layers = 1  # block有几层
        self.factor = 1  # factor是控制注意力稀疏程度的参数，每间隔factor进行采样
        self.dropout = 0.1
        # self.d_ff = 128     # 前馈网络Feed Forward中的隐藏层的大小，也就是第一个全连接层的输出维度

        # gnn and lstm
        self.gnn_in_features = 1
        self.gnn_out_features = 16
        self.gnn_in_timestamps = 128
        # self.lstm_hidden_size = 64
        # self.lstm_out_features = 128

        # TCN features
        self.tcn_layers = [64, 128]
        self.tcn_final_out_channels = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_input_channels = self.input_channels * self.gnn_out_features

        # classifier and discriminator input = final_out_channels * features_len
        self.final_out_channels = 128
        self.label_relation_classifier_input = self.input_channels * self.input_channels
        self.feature_classifier_input = 128
        self.features_len = 1
        self.disc_hid_dim = 64

        # CNN features
        self.cnn_input_channels = 1
        self.mid_channels = 16
        self.stride = 1
        self.cnn_features_len = 1
        self.cnn_kernel_size = 5


class EEG():
    def __init__(self):
        super().__init__()
        self.dataset_name = 'EEG'
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']

        # self.scenarios = []
        # for i in range(0, 19):
        #     for j in range(0, 19):
        #         if i != j:
        #             self.scenarios.append((str(i), str(j)))

        # self.scenarios = [("16", "1")]
        # self.scenarios = [("0", "11"), ("2", "5"), ("12", "5"), ("7", "18"), ("16", "1"),
        #                   ("9", "14"), ("4", "12"), ("10", "7"), ("6", "3"), ("8", "10")]

        self.scenarios = [
            ("0", "11"),
            # ("2", "5"),
            # ("12", "5"),
            # ("7", "18"),
            # ("16", "1"),
            # ("9", "14"),
            # ("4", "12"),
            # ("10", "7"),
            # ("6", "3"),
            # ("8", "10")
        ]


        # base information
        self.sequence_len = 3000
        self.num_classes = 5
        self.input_channels = 1
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # task_fusion
        self.d_model = 128  # 输入transformer的特征维度
        self.n_heads = 3  # 有几个head
        self.e_layers = 1  # block有几层
        self.factor = 1  # factor是控制注意力稀疏程度的参数，每间隔factor进行采样
        self.dropout = 0.1

        # gnn and lstm
        self.gnn_in_features = 1
        self.gnn_out_features = 16
        self.gnn_in_timestamps = 128

        # TCN features
        self.tcn_layers = [75, 128]
        self.tcn_final_out_channels = self.tcn_layers[-1]
        self.tcn_kernel_size = 15
        self.tcn_input_channels = self.input_channels * self.gnn_out_features

        # classifier and discriminator input = final_out_channels * features_len
        self.final_out_channels = 128
        self.label_relation_classifier_input = self.input_channels * self.input_channels
        self.feature_classifier_input = 128
        self.features_len = 1
        self.disc_hid_dim = 64

        # CNN features
        self.cnn_input_channels = 1
        self.mid_channels = 64
        self.stride = 6
        self.cnn_features_len = 1
        self.cnn_kernel_size = 25


class WISDM(object):
    def __init__(self):
        super().__init__()
        self.dataset_name = 'WISDM'
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']

        # self.scenarios = []
        # for i in range(0, 19):
        #     for j in range(0, 19):
        #         if i != j:
        #             self.scenarios.append((str(i), str(j)))

        # self.scenarios = [("12", "7")]
        # self.scenarios = [("2", "32"), ("4", "15"), ("7", "30"), ("12", "17"), ("12", "19"),
        #                   ("18", "20"), ("20", "30"), ("21", "31"), ("25", "29"), ("26", "2")]
        self.scenarios = [
            # ("12", "19"),
            ("12", "7"),
            # ("18", "20"),
            # ("19", "2"),
            # ("2", "28"),
            # ("26", "2"),
            # ("28", "2"),
            # ("28", "20"),
            # ("7", "2"),
            # ("7", "26"),
        ]

        # base information
        self.sequence_len = 128
        self.num_classes = 6
        self.input_channels = 3
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # task_fusion
        self.d_model = 128  # 输入transformer的特征维度
        self.n_heads = 3  # 有几个head
        self.e_layers = 1  # block有几层
        self.factor = 1  # factor是控制注意力稀疏程度的参数，每间隔factor进行采样
        self.dropout = 0.1

        # gnn and lstm
        self.gnn_in_features = 1
        self.gnn_out_features = 16
        self.gnn_in_timestamps = 128

        # TCN features
        self.tcn_layers = [75, 128]
        self.tcn_final_out_channels = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_input_channels = self.input_channels * self.gnn_out_features

        # classifier and discriminator input = final_out_channels * features_len
        self.final_out_channels = 128
        self.label_relation_classifier_input = self.input_channels * self.input_channels
        self.feature_classifier_input = 128
        self.features_len = 1
        self.disc_hid_dim = 64

        # CNN features
        self.cnn_input_channels = 1
        self.mid_channels = 16
        self.stride = 1
        self.cnn_features_len = 1
        self.cnn_kernel_size = 5


class HHAR_SA(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self):
        super().__init__()
        self.dataset_name = 'HHAR_SA'
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']

        # self.scenarios = []
        # for i in range(0, 8):
        #     for j in range(0, 8):
        #         if i != j:
        #             self.scenarios.append((str(i), str(j)))

        # self.scenarios = [("0", "2")]
        # self.scenarios = [("0", "2"), ("1", "6"), ("2", "4"), ("4", "0"), ("4", "5"),
        #                   ("5", "1"), ("5", "2"), ("7", "2"), ("7", "5"), ("8", "4")]

        self.scenarios = [
            # ("0", "2"),
            # ("1", "6"),
            # ("2", "4"),
            # ("4", "0"),
            # ("4", "1"),
            # ("5", "1"),
            # ("7", "1"),
            ("7", "5"),
            # ("8", "3"),
            # ("8", "4")
        ]




        # base information
        self.sequence_len = 128
        self.num_classes = 6
        self.input_channels = 3
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # task_fusion
        self.d_model = 128  # 输入transformer的特征维度
        self.n_heads = 3  # 有几个head
        self.e_layers = 1  # block有几层
        self.factor = 1  # factor是控制注意力稀疏程度的参数，每间隔factor进行采样
        self.dropout = 0.1

        # gnn and lstm
        self.gnn_in_features = 1
        self.gnn_out_features = 16
        self.gnn_in_timestamps = 128

        # TCN features
        self.tcn_layers = [64, 128]
        self.tcn_final_out_channels = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_input_channels = self.input_channels * self.gnn_out_features

        # classifier and discriminator input = final_out_channels * features_len
        self.final_out_channels = 128
        self.label_relation_classifier_input = self.input_channels * self.input_channels
        self.feature_classifier_input = 128
        self.features_len = 1
        self.disc_hid_dim = 64

        # CNN features
        self.cnn_input_channels = 1
        self.mid_channels = 16
        self.stride = 1
        self.cnn_features_len = 1
        self.cnn_kernel_size = 5

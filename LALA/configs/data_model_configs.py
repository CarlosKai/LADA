def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class HAR():
    def __init__(self):
        super(HAR, self)

        # model transformer
        self.d_model = 128  # 输入transformer的特征维度
        self.n_heads = 2    # 有几个multihead

        # patchTST

        self.e_layers = 2   # block有几层
        self.d_ff = 256     # 前馈网络Feed Forward中的隐藏层的大小，也就是第一个全连接层的输出维度
        self.factor = 5     # factor是控制注意力稀疏程度的参数。
        self.activation = "relu"
        self.enc_in = 9     # enc_in 表示输入特征的维度，也可以理解为每个时间步的数据特征数量。

        self.dropout = 0.5

        # domain_discriminator
        # self.domain_discriminator_input = 128

        # self.scenarios = [("1", "2")]
        # self.scenarios = [("2", "11"), ("6", "23"), ("7", "13"), ("9", "18"), ("12", "16"),

        self.scenarios = [
            ("1", "2"), ("1", "3"), ("1", "4"), ("1", "5"), ("1", "6"), ("1", "7"), ("1", "8"), ("1", "9"), ("1", "10"),
            ("2", "3"), ("2", "4"), ("2", "5"), ("2", "6"), ("2", "7"), ("2", "8"), ("2", "9"), ("2", "10"),
            ("3", "4"), ("3", "5"), ("3", "6"), ("3", "7"), ("3", "8"), ("3", "9"), ("3", "10"),
            ("4", "5"), ("4", "6"), ("4", "7"), ("4", "8"), ("4", "9"), ("4", "10"),
            ("5", "6"), ("5", "7"), ("5", "8"), ("5", "9"), ("5", "10"),
            ("6", "7"), ("6", "8"), ("6", "9"), ("6", "10"),
            ("7", "8"), ("7", "9"), ("7", "10"),
            ("8", "9"), ("8", "10"),
            ("9", "10")
        ]

        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.num_classes = 6

        #GAT
        self.in_dim = 128
        self.hidden_dim = 64
        self.out_dim = 128
        self.num_heads = 4

        # self.final_out_channels = 128
        self.features_len = 2


        # TCN features
        self.tcn_layers = [64, 128]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128


        
        
class EEG():
    def __init__(self):
        super(EEG, self).__init__()

        # model transformer
        self.d_model = 256
        self.n_heads = 6

        # patchTST
        self.seq_len = 3000
        self.e_layers = 6
        self.d_ff = 256  # 前馈网络中的隐藏层的大小，也就是第一个全连接层的输出维度
        self.factor = 5
        self.activation = "relu"
        self.enc_in = 1
        self.num_class = 5
        self.patch_len = 64
        self.stride = 32





        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("7", "18"), ("9", "14"), ("12", "5")]
        # self.scenarios = [("0", "11"), ("7", "18"), ("9", "14"), ("12", "5"), ("16", "1"),
        #                   ("3", "19"), ("18", "12"), ("13", "17"), ("5", "15"), ("6", "2")]
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15# 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100


class WISDM(object):
    def __init__(self):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 128
        self.scenarios = [("7", "18"), ("20", "30"), ("35", "31"), ("17", "23"), ("6", "19"),
                          ("2", "11"), ("33", "12"), ("5", "26"), ("28", "4"), ("23", "32")]
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75,150,300]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500


class HHAR(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self):
        super(HHAR, self).__init__()
        self.sequence_len = 128
        self.scenarios = [("0", "6"), ("1", "6"), ("2", "7"), ("3", "8"), ("4", "5"),
                          ("5", "0"), ("6", "1"), ("7", "4"), ("8", "3"), ("0", "2")]
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75,150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        
        
class FD(object):
    def __init__(self):
        super(FD, self).__init__()
        self.sequence_len = 5120
        self.scenarios = [("0", "1"), ("0", "3"), ("1", "0"), ("1", "2"),("1", "3"),
                          ("2", "1"),("2", "3"),  ("3", "0"), ("3", "1"), ("3", "2")]
        self.class_names = ['Healthy', 'D1', 'D2']
        self.num_classes = 3
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # Model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.5

        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

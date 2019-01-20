import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,
                 inp_size,
                 cla_type,
                 cla_size,
                 cla_layers,
                 bidirectional,
                 num_classes,
                 cuda=1):

        super(Model, self).__init__()
        self.t_width = 4

        self.bidirectional = bidirectional
        if bidirectional:
            self.bidirectional = True
        else:
            self.bidirectional = False

        if self.bidirectional:
            self.p_dropout_rnn = 0
            self.rnn_hid_size = 2 * cla_size

        else:
            self.p_dropout_rnn = 0.5
            self.rnn_hid_size = cla_size
        self.fc_size = self.rnn_hid_size
        self.kernel_size = 3
        self.stride = 1
        self.dilation = 1

        # ASR
        self.rnn_type = cla_type
        if cla_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=inp_size, hidden_size=cla_size, num_layers=cla_layers, bias=True,
                                bidirectional=self.bidirectional, dropout=self.p_dropout_rnn)
        elif cla_type == 'GRU':
            self.rnn = nn.GRU(input_size=40, hidden_size=cla_size, num_layers=cla_layers, bias=True, bidirectional=self.bidirectional, dropout=self.p_dropout_rnn)

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.rnn_hid_size, out_features=self.fc_size, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=self.fc_size, out_features=num_classes, bias=True)
        )


    def forward(self, x, show_sp=False):


        self.rnn.flatten_parameters()
        out_rnn, _ = self.rnn(x)
        out_rnn = out_rnn.transpose(0, 1)

        out_fc = self.fc(out_rnn)

        # Select return values
        ctc_out = out_fc

        return ctc_out


class VADModel(nn.Module):
    def __init__(self, n_feature, num_classes):
        super(VADModel, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=n_feature, out_features=256, bias=True),
            nn.RReLU(lower=0),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.RReLU(lower=0),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.RReLU(lower=0),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=num_classes, bias=True)
        )

    def forward(self, x, show_sp=False):
        x = self.fc(x)
        return x

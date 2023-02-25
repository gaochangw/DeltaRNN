import torch
from torch import Tensor
import torch.nn as nn
from project import Project
from utils import util


class Model(nn.Module):
    def __init__(self, proj: Project):
        super(Model, self).__init__()
        # Load Hyperparameters
        for k, v in proj.hparams.items():
            setattr(self, k, v)

        # Debug Dictionary
        self.dict_debug = {} if self.debug else None

        # Instantiate RNN layers
        if self.rnn_type == 'LSTM':
            proj.additem("rnn_layer", "LSTM")
            self.rnn = nn.LSTM(input_size=self.inp_size,
                               hidden_size=self.rnn_size,
                               num_layers=self.rnn_layers,
                               bias=True,
                               bidirectional=False,
                               dropout=self.rnn_dropout)
        elif self.rnn_type == 'GRU':
            proj.additem("rnn_layer", "GRU")
            self.rnn = nn.GRU(input_size=self.inp_size,
                              hidden_size=self.rnn_size,
                              num_layers=self.rnn_layers,
                              bias=True,
                              bidirectional=False,
                              dropout=self.rnn_dropout)
        else:
            raise RuntimeError("Please key in a supported RNN type in the argument.")

        # Extra FC layer after RNN
        if self.fc_extra_size != 0:
            self.fc_extra = nn.Sequential(
                nn.Linear(in_features=self.rnn_size, out_features=self.fc_extra_size, bias=True),
                nn.ReLU(),
                nn.Dropout(p=self.fc_dropout)
            )
            cl_in_features = self.fc_extra_size
        else:
            cl_in_features = self.rnn_size

        # Class Layer
        self.cl = nn.Linear(in_features=cl_in_features, out_features=self.num_classes, bias=True)

        # Initialize Parameters
        self.init_weight()

    def init_weight(self):
        # Initialize LSTM
        for name, param in self.rnn.named_parameters():
            print('::: Initializing Parameters: ', name)
            if 'l0' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param[:self.rnn_size, :])
                    nn.init.xavier_uniform_(param[self.rnn_size:2 * self.rnn_size, :])
                    nn.init.xavier_uniform_(param[2 * self.rnn_size:3 * self.rnn_size, :])
                    nn.init.xavier_uniform_(param[3 * self.rnn_size:, :])
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param[:self.rnn_size, :])
                    nn.init.orthogonal_(param[self.rnn_size:2 * self.rnn_size, :])
                    nn.init.orthogonal_(param[2 * self.rnn_size:3 * self.rnn_size, :])
                    nn.init.orthogonal_(param[3 * self.rnn_size:, :])
            else:
                if 'weight' in name:
                    nn.init.orthogonal_(param[:self.rnn_size, :])
                    nn.init.orthogonal_(param[self.rnn_size:2 * self.rnn_size, :])
                    nn.init.orthogonal_(param[2 * self.rnn_size:3 * self.rnn_size, :])
                    nn.init.orthogonal_(param[3 * self.rnn_size:, :])
            if 'bias' in name:
                nn.init.constant_(param, 0)
        print("--------------------------------------------------------------------")

        # Initialize FC Extra
        if hasattr(self, 'fc_extra'):
            for name, param in self.fc_extra.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                if 'bias' in name:
                    nn.init.constant_(param, 0)

        # Initialize CL
        for name, param in self.cl.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

    def quantize_weight(self, stat):
        for name, param in self.named_parameters():
            # Scale CL layer dynamic range
            if 'cl' in name and self.qcw:
                if stat['net_out_abs_max'] > stat['drange_max']:
                    param.data = param.data / (stat['cl_w_scale'])
                    print("::: Scaling down CL for evaluation. net_out_abs_max={a:f}, drange_max={b:f}".format(
                        a=stat['net_out_abs_max'], b=stat['drange_max']))
            # Quantize Network
            if 'cl' in name and self.qcw:
                param.data = util.quantize_tensor(param.data, self.cwqi, self.cwqf, self.qcw)
                print("::: %s quantized as Q%d.%d" % (name, self.cwqi, self.cwqf))
            elif self.qw:
                param.data = util.quantize_tensor(param.data, self.wqi, self.wqf, self.qw)
                print("::: %s quantized as Q%d.%d" % (name, self.wqi, self.wqf))

    def load_pretrain_model(self, path):
        model_location = next(self.parameters()).device
        pretrained_model_state_dict = torch.load(path, map_location=model_location)
        self.load_state_dict(pretrained_model_state_dict)

    def forward(self, input: Tensor, h_0: Tensor = None):
        # Flatten Parameters
        self.rnn.flatten_parameters()

        # Overhead
        reg = torch.zeros(1).squeeze()
        dict_debug = {}  # Reset Debug Dict
        qa = 0 if self.training else self.qa

        # RNN Forward
        rnn_output, rnn_h_n = self.rnn(input, h_0)
        rnn_output = rnn_output.transpose(0, 1)  # Transpose RNN Output to (N, T, H)

        # FC Forward
        if self.fc_extra_size:
            out_fc = self.fc_extra(rnn_output)
            out_fc = util.quantize_tensor(out_fc, self.aqi, self.aqf, qa)
            out_fc = self.cl(out_fc)
        else:
            out_fc = self.cl(rnn_output)
        out_fc_acc = util.quantize_tensor(out_fc, self.bw_acc, self.aqf + self.wqf, qa)
        qout_fc = util.quantize_tensor(out_fc, self.cqi, self.cqf, self.qc)
        output = qout_fc

        if self.debug:
            dict_debug['fc_final_inp'] = rnn_output
            dict_debug['fc_final_out'] = out_fc
            dict_debug['fc_final_qout'] = qout_fc
            dict_debug['fc_final_out_acc'] = out_fc_acc

        return output, reg

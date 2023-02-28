import torch
from torch import Tensor
import torch.nn as nn
from project import Project
import networks.nn_util as util
from thop import profile
from thop import clever_format

class Model(nn.Module):
    def __init__(self, proj: Project):
        super(Model, self).__init__()
        # Load Hyperparameters
        self.input_size = proj.input_size
        self.batch_size = proj.batch_size
        self.rnn_type = proj.rnn_type
        self.rnn_size = proj.rnn_size
        self.rnn_layers = proj.rnn_layers
        self.rnn_dropout = proj.rnn_dropout
        self.fc_extra_size = proj.fc_extra_size
        self.fc_dropout = proj.fc_dropout
        self.num_classes = proj.num_classes
        self.qa = proj.qa
        self.aqi = proj.aqi
        self.aqf = proj.aqf
        self.qw = proj.qw
        self.wqi = proj.wqi
        self.wqf = proj.wqf
        self.bw_acc = proj.bw_acc
        self.qc = proj.qc
        self.cqi = proj.cqi
        self.cqf = proj.cqf
        self.qcw = proj.qcw
        self.cwqi = proj.cwqi
        self.cwqf = proj.cwqf
        self.debug = proj.debug

        # Debug Dictionary
        self.statistics = {}

        # Instantiate RNN layers
        if self.rnn_type == 'LSTM':
            proj.additem("rnn_layer", "LSTM")
            self.rnn = nn.LSTM(input_size=self.input_size,
                               hidden_size=self.rnn_size,
                               num_layers=self.rnn_layers,
                               bias=True,
                               bidirectional=False,
                               dropout=self.rnn_dropout)
        elif self.rnn_type == 'GRU':
            proj.additem("rnn_layer", "GRU")
            self.rnn = nn.GRU(input_size=self.input_size,
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
        self.reset_parameters()

    def reset_parameters(self):
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

    def get_model_size(self):
        # device = next(self.parameters()).device
        # seq_len = 1
        # input = torch.randn(seq_len, self.batch_size, self.input_size, device=device)
        # macs, params = profile(self, inputs=(input,))
        # macs, params = clever_format([macs, params], "%.3f")
        # self.statistics['MACS'] = macs
        # self.statistics['PARAMS'] = params
        self.statistics['NUM_PARAMS'] = 0
        for name, param in self.named_parameters():
            self.statistics['NUM_PARAMS'] += param.data.numel()
        return self.statistics['NUM_PARAMS']

    def get_sparsity(self):
        for name, module in self._modules.items():
            self.statistics['SP_W_' + name.upper()] = util.get_layer_sparsity(module)
        return self.statistics

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

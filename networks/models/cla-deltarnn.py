import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import networks.nn_util as util
from networks.layers.deltagru import DeltaGRU
from networks.layers.deltalstm import DeltaLSTM
from networks.layers.deltalstm3 import DeltaLSTM3


class Model(nn.Module):
    def __init__(self, proj):
        super(Model, self).__init__()
        # Load Hyperparameters
        self.input_size = proj.input_size
        self.batch_size = proj.batch_size
        self.rnn_type = proj.rnn_type
        self.rnn_size = proj.rnn_size
        self.rnn_layers = proj.rnn_layers
        self.rnn_dropout = proj.rnn_dropout
        self.thx = proj.thx
        self.thh = proj.thh
        self.use_hardtanh = proj.use_hardtanh
        self.use_hardsigmoid = proj.use_hardsigmoid
        self.fc_extra_size = proj.fc_extra_size
        self.fc_dropout = proj.fc_dropout
        self.num_classes = proj.num_classes
        self.qa = proj.qa
        self.aqi = proj.aqi
        self.aqf = proj.aqf
        self.qw = proj.qw
        self.wqi = proj.wqi
        self.wqf = proj.wqf
        self.nqi = proj.nqi
        self.nqf = proj.nqf
        self.bw_acc = proj.bw_acc
        self.qc = proj.qc
        self.cqi = proj.cqi
        self.cqf = proj.cqf
        self.qcw = proj.qcw
        self.cwqi = proj.cwqi
        self.cwqf = proj.cwqf
        self.debug = proj.debug

        # Statistics Dictionary
        self.statistics = {}

        # Instantiate RNN layers
        if self.rnn_type == 'LSTM':
            proj.additem("rnn_layer", "DeltaLSTM")
            self.rnn = DeltaLSTM(input_size=self.input_size,
                                 hidden_size=self.rnn_size,
                                 num_layers=self.rnn_layers,
                                 thx=self.thx,
                                 thh=self.thh,
                                 qa=self.qa,
                                 aqi=self.aqi,
                                 aqf=self.aqf,
                                 qw=self.qw,
                                 wqi=self.wqi,
                                 wqf=self.wqf,
                                 nqi=self.nqi,
                                 nqf=self.nqf,
                                 bw_acc=self.bw_acc,
                                 use_hardtanh=self.use_hardtanh,
                                 use_hardsigmoid=self.use_hardsigmoid,
                                 debug=self.debug
                                 )
        elif self.rnn_type == 'LSTM3':
            proj.additem("rnn_layer", "DeltaLSTM3")
            self.rnn = DeltaLSTM3(input_size=self.input_size,
                                  hidden_size=self.rnn_size,
                                  num_layers=self.rnn_layers,
                                  thx=self.thx,
                                  thh=self.thh,
                                  # dropout=self.dropout
                                  qa=self.qa,
                                  aqi=self.aqi,
                                  aqf=self.aqf,
                                  qw=self.qw,
                                  wqi=self.wqi,
                                  wqf=self.wqf,
                                  nqi=self.nqi,
                                  nqf=self.nqf,
                                  # qg=self.qg,
                                  # gqi=self.gqi,
                                  # gqf=self.gqf,
                                  bw_acc=self.bw_acc,
                                  use_hardtanh=self.use_hardtanh,
                                  use_hardsigmoid=self.use_hardsigmoid,
                                  debug=self.debug
                                  )
        elif self.rnn_type == 'GRU':
            proj.additem("rnn_layer", "DeltaGRU")
            self.rnn = DeltaGRU(input_size=self.input_size,
                                hidden_size=self.rnn_size,
                                num_layers=self.rnn_layers,
                                thx=self.thx,
                                thh=self.thh,
                                qa=self.qa,
                                aqi=self.aqi,
                                aqf=self.aqf,
                                qw=self.qw,
                                wqi=self.wqi,
                                wqf=self.wqf,
                                nqi=self.nqi,
                                nqf=self.nqf,
                                bw_acc=self.bw_acc,
                                use_hardtanh=self.use_hardtanh,
                                use_hardsigmoid=self.use_hardsigmoid,
                                debug=self.debug
                                )
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

    def set_debug(self, value):
        setattr(self, "debug", value)
        self.rnn.set_debug(value)

    def reset_parameters(self):
        # Initialize DeltaLSTM
        self.rnn.reset_parameters()

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
                    print("###Scaling down FC Final for evaluation. net_out_abs_max={a:f}, drange_max={b:f}".format(
                        a=stat['net_out_abs_max'], b=stat['drange_max']))
            # Quantize Network
            if 'cl' in name and self.qcw:
                param.data = util.quantize_tensor(param.data, self.cwqi, self.cwqf, self.qcw)
                print("::: %s quantized as Q%d.%d" % (name, self.cwqi, self.cwqf))
            elif self.qw:
                param.data = util.quantize_tensor(param.data, self.wqi, self.wqf, self.qw)
                print("::: %s quantized as Q%d.%d" % (name, self.wqi, self.wqf))

    def column_balanced_targeted_dropout(self, alpha):
        for name, param in self.named_parameters():
            if 'fc_extra' in name:
                if 'weight' in name:
                    cbtd(param.data, gamma=self.gamma_fc, alpha=alpha, num_pe=self.num_array_pe)
            if 'rnn' in name:
                if 'weight' in name:
                    cbtd(param.data, gamma=self.gamma_rnn, alpha=alpha, num_pe=self.num_array_pe)
                    print("::: %s pruned by CBTD.", name)

    def load_pretrain_model(self, path):
        model_location = next(self.parameters()).device
        pretrained_model_state_dict = torch.load(path, map_location=model_location)
        self.load_state_dict(pretrained_model_state_dict)

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
        temporal_sparsity = self.rnn.get_temporal_sparsity()
        self.statistics.update(temporal_sparsity)
        return self.statistics

    def forward(self, input: Tensor, x_p_0: Tensor = None, h_0: Tensor = None, h_p_0: Tensor = None,
                c_0: Tensor = None, dm_0: Tensor = None):
        # Flatten Parameters
        self.rnn.flatten_parameters()

        # Overhead
        self.statistics = {}  # Reset Debug Dict
        qa = 0 if self.training else self.qa

        # RNN Forward
        rnn_output, (x_p_n, h_n, h_p_n, c_n, dm_n), reg = self.rnn(input, x_p_0, h_0, h_p_0, c_0, dm_0)
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
        # outputs = (qout_fc, rnn_output)
        output = qout_fc

        if self.debug:
            self.statistics['fc_final_inp'] = rnn_output
            self.statistics['fc_final_out'] = out_fc
            self.statistics['fc_final_qout'] = qout_fc
            self.statistics['fc_final_out_acc'] = out_fc_acc
            self.statistics.update(self.rnn.statistics)

        return output, reg


def cbtd(x, gamma, alpha, num_pe):
    """
    :param x: input tensor (weight matrix)
    :param gamma: target sparsity
    :param alpha: drop probability
    :param num_pe: number of processing elements along the column direction (or number of submatrices)
    :return: tensor quantized to fixed-point precision
    """
    import math

    n_dims = len(x.size())
    n_rows = x.size(0)

    if n_dims == 1:
        n_cols = 1
    elif n_dims > 1:
        n_cols = x.size(1)
    else:
        raise RuntimeError('Input tensor must have at least 1 dimension.')

    # Split and shuffle weight matrix
    # for i in tqdm(range(0, num_pe), total=num_pe, unit="Submat"):
    for i in range(0, num_pe):
        # Vector
        if n_dims == 1:
            pe_work = x[np.arange(i, n_rows, num_pe)]
            pe_work_abs = torch.abs(pe_work)
            drop_part = math.floor(pe_work_abs.shape[0] * gamma)
            _, indices = torch.sort(pe_work_abs, dim=0)
            drop_indices = indices[0:drop_part]
            drop_rand_mask = torch.rand(drop_indices.size(0), device=x.device)
            pe_work[drop_indices] = pe_work[drop_indices].masked_fill_(drop_rand_mask <= alpha, 0)
            x[np.arange(i, n_rows, num_pe)] = pe_work
        # Matrix
        elif n_dims > 1:
            pe_work = x[np.arange(i, n_rows, num_pe), :]
            pe_work_abs = torch.abs(pe_work)
            drop_part = math.floor(pe_work_abs.shape[0] * gamma)
            _, indices = torch.sort(pe_work_abs, dim=0)
            drop_indices = indices[0:drop_part, :]
            drop_rand_mask = torch.rand(drop_indices.size(0), drop_indices.size(1), device=x.device)
            for j in range(0, n_cols):
                pe_work[drop_indices[:, j], j] = pe_work[drop_indices[:, j], j].masked_fill_(
                    drop_rand_mask[:, j] <= alpha, 0)
            x[np.arange(i, n_rows, num_pe), :] = pe_work
    return x

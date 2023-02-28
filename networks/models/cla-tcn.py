import torch
from torch import Tensor
import torch.nn as nn
from networks.layers.tcn1d import TemporalConvNet1D
import networks.nn_util as util
from project import Project
from thop import profile
from thop import clever_format

class Model(nn.Module):
    def __init__(self, proj: Project):
        super(Model, self).__init__()
        # Load Hyperparameters
        self.input_size = proj.input_size
        self.batch_size = proj.batch_size
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
        self.tcn_num_channels = [150]*4

        # Statistics Dictionary
        self.statistics = {}

        # Instantiate RNN layers
        self.cnn = TemporalConvNet1D(num_inputs=self.input_size, num_channels=self.tcn_num_channels)

        # Extra FC layer after RNN
        self.tcn_output_size = self.tcn_num_channels[-1]
        if self.fc_extra_size != 0:
            self.fc_extra = nn.Sequential(
                nn.Linear(in_features=self.tcn_output_size, out_features=self.fc_extra_size, bias=True),
                nn.ReLU(),
                nn.Dropout(p=self.fc_dropout)
            )
            cl_in_features = self.fc_extra_size
        else:
            cl_in_features = self.tcn_output_size

        # Class Layer
        self.cl = nn.Linear(in_features=cl_in_features, out_features=self.num_classes, bias=True)

        # Initialize Parameters
        self.reset_parameters()

    def reset_parameters(self):
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

    def get_sparsity(self):
        for name, module in self._modules.items():
            self.statistics['SP_W_' + name.upper()] = util.get_layer_sparsity(module)
        return self.statistics

    def get_model_size(self):
        device = next(self.parameters()).device
        seq_len = 1
        input = torch.randn(seq_len, self.batch_size, self.input_size, device=device)
        macs, params = profile(self, inputs=(input,))
        macs, params = clever_format([macs, params], "%.3f")
        self.statistics['MACS'] = macs
        self.statistics['PARAMS'] = params
        return macs, params

    def load_pretrain_model(self, path):
        model_location = next(self.parameters()).device
        pretrained_model_state_dict = torch.load(path, map_location=model_location)
        self.load_state_dict(pretrained_model_state_dict)

    def forward(self, input: Tensor):
        # Overhead
        reg = torch.zeros(1).squeeze()
        dict_debug = {}  # Reset Debug Dict
        qa = 0 if self.training else self.qa

        # TCN Forward
        feat = torch.permute(input, (1, 2, 0))
        # residual = feat
        out_tcn = self.cnn(feat)
        out_tcn = torch.transpose(out_tcn, 1, 2)

        # FC Forward
        if self.fc_extra_size:
            out_fc = self.fc_extra(out_tcn)
            out_fc = util.quantize_tensor(out_fc, self.aqi, self.aqf, qa)
            out_fc = self.cl(out_fc)
        else:
            out_fc = self.cl(out_tcn)
        out_fc_acc = util.quantize_tensor(out_fc, self.bw_acc, self.aqf + self.wqf, qa)
        qout_fc = util.quantize_tensor(out_fc, self.cqi, self.cqf, self.qc)
        output = qout_fc

        return output, reg

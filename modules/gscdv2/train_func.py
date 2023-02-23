import torch
import torch.nn as nn
import utils.util as util
import numpy as np


def forward_propagation(net, dict_batch_tensor, state=None):
    output, reg = net(dict_batch_tensor['features'])
    return output, reg


def get_batch_data(proj, batch):
    # Get Data
    features, feature_lengths, targets, target_lengths, flags = batch
    # Fetch Batch Data
    # features = features.transpose(0, 1)  # Dim: (T, N, F)
    targets_metric = targets  # Labels for scoring

    if proj.use_cuda:
        features = features.cuda()
        targets = targets.cuda()

    dict_batch_tensor = {'features': features,
                         'targets': targets,
                         'targets_metric': targets_metric,
                         'flags': flags,
                         'feature_lengths': feature_lengths,
                         'target_lengths': target_lengths}
    return dict_batch_tensor


def calculate_loss(loss_fn, net_out, dict_targets, reg, beta):
    selected_net_out = net_out
    selected_targets = dict_targets['targets']
    loss_fn_input = selected_net_out.reshape((-1, selected_net_out.size(-1)))
    loss_fn_target = selected_targets.reshape((-1))
    loss = loss_fn(loss_fn_input, loss_fn_target)
    if reg is not None:
        loss_reg = reg * beta
        loss += loss_reg
    else:
        loss_reg = 0
    return loss, loss_reg


def add_meter_data(meter, dict_meter_data):
    meter.extend_data(outputs=dict_meter_data['net_qout'],
                      targets=dict_meter_data['targets_metric'],
                      flags=dict_meter_data['flag'])
    return meter


def process_network(proj, stat, alpha):
    proj.net.quantize_weight(stat)
    if proj.cbtd:
        proj.net.column_balanced_targeted_dropout(alpha)


def get_net_out_stat(proj, stat: dict, meter_data: dict):
    # Get max(abs(x)) of network outputs
    net_out_ravel = []
    for batch in meter_data['net_out']:
        net_out_ravel.append(batch.view(-1))
    net_out_ravel = torch.cat(net_out_ravel)
    stat['net_out_max'] = torch.max(net_out_ravel).item()
    stat['net_out_min'] = torch.min(net_out_ravel).item()
    stat['net_out_abs_max'] = torch.max(torch.abs(net_out_ravel)).item()

    # Get max(abs(x)) of network quantized outputs
    net_qout_ravel = []
    for batch in meter_data['net_qout']:
        net_qout_ravel.append(batch.view(-1))
    net_qout_ravel = torch.cat(net_qout_ravel)
    stat['net_qout_max'] = torch.max(net_qout_ravel).item()
    stat['net_qout_min'] = torch.min(net_qout_ravel).item()
    stat['net_qout_abs_max'] = torch.max(torch.abs(net_qout_ravel)).item()

    # Get dynamic range of final layer quantization
    stat['drange_max'] = float(2 ** (proj.cqi + proj.cqf - 1) - 1) / float(
        2 ** proj.cqf)
    # drange_min = -float(2 ** (proj.aqi_cl + proj.aqi_cl - 1)) / float(
    #     2 ** proj.aqi_cl)
    # qstep = 1 / float(2 ** proj.aqi_cl)

    # Get final FC layer weight scale factor
    if stat['net_out_abs_max'] > stat['drange_max']:
        stat['cl_w_scale'] = stat['net_out_abs_max'] / stat['drange_max']
    return stat


def initialize_network(proj, net):
    print('::: Initializing Parameters:')
    hid_size = proj.rnn_size
    hid_type = proj.rnn_type
    for name, param in net.named_parameters():
        print(name)
        # qLSTM uses its own initializer including quantization
        if 'rnn' in name and hid_type not in ['qLSTM']:
            num_gates = int(param.shape[0] / hid_size)
            if 'bias' in name:
                nn.init.constant_(param, 0)
            if 'weight' in name:
                for i in range(0, num_gates):
                    nn.init.orthogonal_(param[i * hid_size:(i + 1) * hid_size, :])
                # nn.init.xavier_normal_(param)
            if 'weight_ih_l0' in name:
                for i in range(0, num_gates):
                    nn.init.xavier_uniform_(param[i * hid_size:(i + 1) * hid_size, :])

        # qLinear uses its own initializer including quantization
        if 'fc' in name:
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                # nn.init.uniform_(param)
                nn.init.constant_(param, 0)
        # if 'bias' in name:  # all biases
        #     nn.init.constant_(param, 0)
        if proj.rnn_type == 'LSTM':  # only LSTM biases
            if ('bias_ih' in name) or ('bias_hh' in name):
                no4 = int(len(param) / 4)
                no2 = int(len(param) / 2)
                nn.init.constant_(param, 0)
                nn.init.constant_(param[no4:no2], 1)
    print("--------------------------------------------------------------------")

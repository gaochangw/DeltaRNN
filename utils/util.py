import collections
import torch
import numpy as np
import os
import h5py
import time
import math
import platform
import typing
from torch import Tensor


def count_net_params(net):
    n_param = 0
    for name, param in net.named_parameters():
        layer_param = torch.numel(param)
        print("::: Layer Name: ", name, "Layer Size", layer_param)
        n_param += layer_param
    return n_param


def count_rnn_params(net):
    n_param = 0
    for name, param in net.named_parameters():
        if 'rnn' in name:
            sizes = 1
            for el in param.size():
                sizes = sizes * el
            n_param += sizes
    return n_param


def get_dataset_path(args):
    # Dictionary for datasets
    if args.dataset_name == 'gscdv1':
        print("Preparing Google Speech Command Dataset...")
        dataset_name = 'speech_commands_v0.01'
        testset_name = 'speech_commands_test_set_v0.01'
    elif args.dataset_name == 'gscdv2':
        print("Preparing Google Speech Command Dataset...")
        dataset_name = 'speech_commands_v0.02'
        testset_name = 'speech_commands_test_set_v0.02'
    elif args.dataset_name == 'timit':
        print("Preparing TIMIT Dataset...")
        dataset_name = 'TIMIT/TIMIT'
        testset_name = None
    elif args.dataset_name == 'tidigits':
        print("Preparing TIDIGITS Dataset...")
        dataset_name = 'tidigits'
        testset_name = None
    else:
        raise RuntimeError('Dataset not supported.')

    data_dir = args.data_dir
    dataset_path = os.path.join(data_dir, dataset_name)
    testset_path = os.path.join(data_dir, testset_name)

    return dataset_path, testset_path


def load_h5py_data(f):
    dict_data = {}
    for k in f.keys():
        dict_data[k] = np.array(f[k])
    return dict_data


def log_lut(x, qi_in, qf_in, qi_out, qf_out):
    lut_in = quantize_array(x, qi_in, qf_in, 1)
    lut_out = np.log10(lut_in + 1)
    lut_out = quantize_array(lut_out, qi_out, qf_out, 1)
    return lut_out


def gen_description_file(augment_noise, target_snr):
    if augment_noise:
        description_file = 'description_snr' + str(target_snr) + '.csv'
    else:
        description_file = 'description.csv'
    return description_file


def piecewise_linear_log(x,
                         k0, b0,
                         k1, b1,
                         k2, b2,
                         k3, b3,
                         k4, b4,
                         k5, b5,
                         k6, b6,
                         k7, b7,
                         k8, b8,
                         k9, b9
                         ):
    return np.piecewise(x=x,
                        condlist=[x >= 0,
                                  x >= 6,
                                  x >= 12,
                                  x >= 18,
                                  x >= 24,
                                  x >= 108,
                                  x >= 192,
                                  x >= 276,
                                  x >= 1549,
                                  x >= 2792],
                        funclist=[lambda x: k0 * x + b0,
                                  lambda x: k1 * x + b1,
                                  lambda x: k2 * x + b2,
                                  lambda x: k3 * x + b3,
                                  lambda x: k4 * x + b4,
                                  lambda x: k5 * x + b5,
                                  lambda x: k6 * x + b6,
                                  lambda x: k7 * x + b7,
                                  lambda x: k8 * x + b8,
                                  lambda x: k9 * x + b9]
                        )


def create_folder(folder_list):
    for folder in folder_list:
        try:
            os.makedirs(folder)
        except:
            pass


def load_model(proj, net, model_path):
    if proj.use_cuda and torch.cuda.is_available():
        model_location = 'cuda:' + str(proj.gpu_device)
    else:
        model_location = 'cpu'
    pretrained_model = torch.load(model_path, map_location=model_location)
    dict_pretrained_model = dict(pretrained_model.items())
    list_pretrained_model = list(dict_pretrained_model.items())
    dict_converted_model = {}
    net_state_dict = net.state_dict()
    list_trained_model = list(net_state_dict.items())
    num_pretrain_keys = len(list_pretrained_model)
    num_train_keys = len(list_trained_model)

    # Convert Model
    print("num_pretrain_keys: ", num_pretrain_keys)
    print("num_train_keys:    ", num_train_keys)
    if num_pretrain_keys != num_train_keys:
        for k_model, v_model in dict_pretrained_model.items():
            if 'ih' in k_model:
                if 'weight' in k_model:
                    param_ih = v_model
                    continue
                if 'bias' in k_model:
                    param_ih = v_model
                    continue
            elif 'hh' in k_model:
                if 'weight' in k_model:
                    param_hh = v_model
                    param_cat = torch.cat((param_ih, param_hh), dim=-1)
                if 'bias' in k_model:
                    param_hh = v_model
                    param_cat = param_ih + param_hh
                dict_converted_model[k_model] = param_cat
            else:
                dict_converted_model[k_model] = v_model
        list_converted_model_params = list(dict_converted_model.items())
    else:
        list_converted_model_params = list_pretrained_model
    # Copy param of the pretrained model to the training network
    new_state_dict = collections.OrderedDict()
    count = 0
    for key, value in net.state_dict().items():
        layer_name, weights = list_converted_model_params[count]
        net_state_dict[key] = weights
        print("Pre-trained Layer: %s - Loaded into new layer: %s" % (layer_name, key))
        count += 1

    net.load_state_dict(net_state_dict)
    return net


# def load_model(net, model_path, gpu_device):
#     # Get adapted DeltaRNN dimensions
#     # adapted_input_size = int(math.ceil(input_size / float(num_array)) * num_array)
#     # adapted_hidden_size = int(math.ceil(hidden_size / float(num_array)) * num_array)
#     # Load pretrained model
#     pretrained_net = torch.load(model_path, map_location='cuda:' + str(gpu_device))
#     pretrained_net = list(pretrained_net.items())
#
#     # Iterate throughput retrain net parameters to load pretrained net parameters correctly
#     new_state_dict = collections.OrderedDict()
#     count = 0
#     num_param_key = len(pretrained_net)
#     for key, value in net.state_dict().items():
#         if count >= num_param_key:
#             break
#         layer_name, weights = pretrained_net[count]
#         new_state_dict[key] = weights
#         print("Pre-trained Layer: %s - Loaded into new layer: %s" % (layer_name, key))
#         count += 1
#     net.load_state_dict(new_state_dict)
#     return net


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def create_context_window(tensor, window_size=2):
    """
    Create a new tensor with context window
    :param tensor: 3D tensor (T, N, F)
    :param window_size: Integer
    :return:
    """
    tensor = tensor.transpose(0, 1)
    dim_T = tensor.shape[0]
    dim_N = tensor.shape[1]
    dim_F = tensor.shape[2]
    dim_FC = int(dim_F / 3)
    dim_C = int(window_size * 3)
    pad_len = window_size - 1
    pad = torch.zeros((pad_len, dim_N, dim_F)).cuda()
    padded_dim_T = dim_T + pad_len
    feature = torch.cat((tensor, pad), dim=0)
    feature = torch.reshape(feature, (padded_dim_T, dim_N, 3, dim_FC))
    output = []
    for tstep in range(0, dim_T):
        temp = feature[tstep:tstep + window_size, :, :, :]
        # print(temp.shape)
        temp = temp.transpose(0, 1)
        temp = torch.reshape(temp, (dim_N, dim_C, dim_FC))
        output.append(temp)
    output = torch.stack(output)
    return output, dim_C, dim_FC


def write_hdf5_normal(filename, X, y):
    # fun
    feature_lens = np.asarray([sample.shape[1] for sample in X]).astype(np.float32)
    label_lens = np.asarray([len(target) for target in y]).astype(np.float32)
    features = np.concatenate(X, axis=1).T.astype(np.float32)
    labels = np.concatenate(y).astype(np.float32)

    with h5py.File(filename, 'w') as f:
        f.create_dataset('features', data=features)
        f.create_dataset('feature_lens', data=feature_lens)
        f.create_dataset('labels', data=labels)
        f.create_dataset('label_lens', data=label_lens)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if indices.nelement() == 0:  # if all elements are zeros
        print("1", indices)
        return sparse_tensortype(*x.shape)
    else:
        print("2", indices)
        indices = indices.torch()
        values = x[tuple(indices[i] for i in range(indices.shape[0]))]
        return sparse_tensortype(indices, values, x.size())


def quantize_tensor(x: Tensor, qi: int, qf: int, en: int = 0, floor=False):
    """
    :param x: input tensor
    :param qi: number of integer bits before the decimal point
    :param qf: number of fraction bits after the decimal point
    :param en: Type of Quantziation (0 - none, 1 - round, 2 - floor)
    :param floor: Whether use floor() instead of round()
    :return: tensor quantized to fixed-point precision
    """
    if en == 0:
        return x
    else:
        power = float(2. ** qf)
        clip_val = float(2. ** (qi + qf - 1))
        if floor:
            value = torch.floor(x * power)
        else:
            value = torch.round(x * power)
        value = torch.clamp(value, -clip_val, clip_val - 1)  # saturation arithmetic
        value = value / power
        return value


def quantize_array(x, qi, qf, enable, unsigned=0, floor=False):
    """
    :param x: input numpy array
    :param qi: number of integer bits before the decimal point
    :param qf: number of fraction bits after the decimal point
    :param enable: whether enable the function
    :param unsigned: whether unsigned input
    :return: array quantized to fixed-point precision
    """
    if enable == 0:
        return x
    power = np.asarray(2. ** qf).astype(np.float64)
    if unsigned == 0:
        clip_val = float(2. ** (qi + qf - 1))
        if floor:
            value = np.floor(x * power)
        else:
            value = np.round(x * power)
        value = np.clip(value, -clip_val, clip_val - 1)  # saturation arithmetic
        result = value / power
    else:
        clip_val = float(2. ** (qi + qf))
        if floor:
            value = np.floor(x * power)
        else:
            value = np.round(x * power)
        value = np.clip(value, 0, clip_val - 1)  # saturation arithmetic
        result = value / power
    return result


# Auto determine minimal possible qi
def quantize_tensor_auto(x, bw, enable):
    if enable == 0:
        return x, 0, 0
    qi = torch.ceil(torch.log2(x)) + 1
    qf = bw - qi
    # if qi < 0 or qf < 0:
    #     print('Quantizing {} to Q({}, {})!'.format(x, qi, qf))
    power = float(2. ** qf)
    clip_val = float(2. ** (qi + qf - 1) - 1)
    value = torch.round(x * power)
    value = torch.clamp(value, -clip_val, clip_val)  # saturation arithmetic
    value = value / power
    return value, qi, qf


def quantize_rnn(net, qi, qf, enable):
    for name, param in net.named_parameters():
        if 'rnn' in name and 'swap' not in name:
            # print("Quantizing Parameter: ", name)
            param.features = quantize_tensor(param.features, qi, qf, enable)
    return net


def quantize_net(net, qi, qf, enable):
    for name, param in net.named_parameters():
        param.features = quantize_tensor(param.features, qi, qf, enable)
    return net


def hard_quantize_tensor(x, m, n, enable=0):
    """
    :param x: input tensor
    :param m: number of integer bits before the decimal point
    :param n: number of fraction bits after the decimal point
    :return: tensor quantized to fixed-point precision
    """

    if not enable:
        return x

    power = 2. ** n
    max_val = 2. ** (m + n - 1) - 1
    value = x * power
    value = torch.floor(value)  # rounding
    value = torch.clamp(value, -max_val, max_val)  # saturation arithmetic
    value = value / power
    return value


def get_temporal_sparsity(list_layer, seq_len, threshold):
    # Evaluate Sparsity
    num_zeros = 0
    num_elems = 0
    # print(seq_len.size())
    # Iterate through layers
    for layer in list_layer:
        all_delta_vec = layer.transpose(0, 1)
        all_delta_vec = torch.abs(all_delta_vec)  # Take absolute values of all delta vector elements
        for i, delta_vec in enumerate(all_delta_vec):
            seq = delta_vec[:seq_len[i], :]
            zero_mask = seq < threshold
            num_zeros += torch.sum(zero_mask)
            num_elems += torch.numel(zero_mask)
    sparsity = float(num_zeros) / float(num_elems)
    return sparsity


# def targeted_dropout(x, gamma, alpha, epoch):
#     """
#     :param x: input tensor
#     :param m: number of integer bits before the decimal point
#     :param n: number of fraction bits after the decimal point
#     :return: tensor quantized to fixed-point precision
#     """
#     import math
#     torch.manual_seed(epoch)
#     torch.cuda.manual_seed_all(epoch)
#     n_dims = len(x.size())
#     n_elements = x.numel()
#     drop_part = math.ceil(n_elements * gamma)
#     weight_vec = x.view(-1)
#     weight_vec_abs = torch.abs(weight_vec)
#     sorted, indices = torch.sort(weight_vec_abs)
#     # print(sorted)
#     drop_indices = indices[0:drop_part]
#     drop_rand_mask = torch.rand(drop_indices.size(0)).cuda()
#     drop_mask = torch.ones(drop_indices.size(0)).cuda()
#     drop_mask = drop_mask.masked_fill_(drop_rand_mask <= alpha, 0)
#     weight_vec[drop_indices] *= drop_mask
#     if n_dims == 1:
#         weight = weight_vec
#     elif n_dims > 1:
#         weight = torch.reshape(weight_vec, (x.size(0), x.size(1)))
#
#     return weight

def aligned_cbwdrop(x, gamma, alpha, num_pe):
    """
    :param x: input tensor (weight matrix)
    :param m: number of integer bits before the decimal point
    :param n: number of fraction bits after the decimal point
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
        if n_dims == 1:
            pe_work = x[np.arange(i, n_rows, num_pe)]
            pe_work_abs = torch.abs(pe_work)
            drop_part = math.floor(pe_work_abs.shape[0] * gamma)
            _, indices = torch.sort(pe_work_abs, dim=0)
            drop_indices = indices[0:drop_part]
            drop_rand_mask = torch.rand(drop_indices.size(0)).cuda()
            pe_work[drop_indices] = pe_work[drop_indices].masked_fill_(drop_rand_mask <= alpha, 0)
            x[np.arange(i, n_rows, num_pe)] = pe_work
        elif n_dims > 1:
            pe_work = x[np.arange(i, n_rows, num_pe), :]
            pe_work_abs = torch.abs(pe_work)
            drop_part = math.floor(pe_work_abs.shape[0] * gamma)
            _, indices = torch.sort(pe_work_abs, dim=0)
            drop_indices = indices[0:drop_part, :]
            drop_rand_mask = torch.rand(drop_indices.size(0), drop_indices.size(1)).cuda()
            for j in range(0, n_cols):
                pe_work[drop_indices[:, j], j] = pe_work[drop_indices[:, j], j].masked_fill_(
                    drop_rand_mask[:, j] <= alpha, 0)
            x[np.arange(i, n_rows, num_pe), :] = pe_work
    return x





# def cbtd(x, gamma, alpha):
#     """
#     x: input tensor
#     gamma: target sparsity
#     alpha: drop probability
#     mode: 0 - normal; 1 -
#     """
#     import math
#     device = x.device
#     n_dims = len(x.size())
#     n_rows = x.size(0)
#
#     if n_dims == 1:
#         n_cols = 1
#     elif n_dims == 2:
#         n_cols = x.size(1)
#     else:
#         raise RuntimeError('Input tensor must have 1 or 2 dimensions.')
#
#     x_abs = torch.abs(x)
#     drop_part = math.floor(n_rows * gamma)
#     _, indices = torch.sort(x_abs, dim=0)
#     if n_dims == 1:
#         drop_indices = indices[0:drop_part]
#         drop_rand_mask = torch.rand(drop_indices.size(0)).cuda()
#         x[drop_indices] = x[drop_indices].masked_fill_(drop_rand_mask <= alpha, 0)
#     elif n_dims > 1:
#         drop_indices = indices[0:drop_part, :]
#         drop_rand_mask = torch.rand(drop_indices.size(0), drop_indices.size(1), device=device)
#         for j in range(0, n_cols):
#             x[drop_indices[:, j], j] = x[drop_indices[:, j], j].masked_fill_(
#                 drop_rand_mask[:, j] <= alpha, 0)
#     return x


def castToFixed(x, m, n):
    """
    :param x: input numpy array
    :param m: number of integer bits before the decimal point
    :param n: number of fraction bits after the decimal point
    :return: tensor quantized to fixed-point precision
    """
    power = 2. ** n
    max_val = 2. ** (m + n) - 1
    value = x * power
    value = np.round(value)  # rounding
    value = np.clip(value, -max_val - 1, max_val)  # saturation arithmetic
    return value


def interleave(x, width):
    """
    PyTorch Function to interleave a
    :param x:
    :param width:
    :return:
    """

    x_dim0 = x.size(0)
    num_dim = len(x.size())
    interleaved = []
    for i in range(0, width):
        idx_curr = torch.arange(i, x_dim0, width).long()
        if num_dim == 1:
            interleaved.append(x[idx_curr])
        elif num_dim == 2:
            interleaved.append(x[idx_curr, :])
    interleaved = torch.stack(interleaved)
    return interleaved


def get_array_work(workload, num_array_pe, num_array):
    n_col = workload.size(0)
    hw_n_col = int(num_array_pe * np.ceil(float(n_col) / float(num_array_pe)))
    num_pad_col = int(hw_n_col - n_col)
    workload_pad_col = torch.zeros(num_pad_col)
    act_workload = torch.cat((workload, workload_pad_col), dim=0)

    array_col_work = torch.zeros(int(num_array), int(hw_n_col / num_array))
    array_work = torch.zeros(int(num_array))
    for i in range(0, int(num_array)):
        array_col_work[i] = act_workload[np.arange(i, hw_n_col, num_array)]
        array_work[i] = torch.sum(array_col_work[i])

    return array_work, array_col_work


def cast_fp_to_fxp(x, m, n):
    """
    Converts a floating point number into a fixed point number

    :param x: input numpy array
    :param m: number of integer bits before the decimal point
    :param n: number of fraction bits after the decimal point
    :return: tensor quantized to fixed-point precision
    """
    power = 2. ** n
    max_val = 2. ** (m + n)
    value = x * power
    value = np.round(value)  # rounding
    value = np.clip(value, -max_val, max_val - 1).astype(np.int32)  # saturation arithmetic
    return value


def cast_fxp_to_hex(x, data_type):
    x = np.asarray(x, dtype=data_type)
    return x


# def cast_signed_to_unsigned(x, bw):
#     temp = x
#     idx_below_zero = np.argwhere(temp<0)
#     abs_below_zero = np.abs(temp[idx_below_zero])
#     max_val = 2**bw
#     unsigned_temp = max_val - abs_below_zero
#     temp[idx_below_zero] = unsigned_temp
#     return temp

def cast_signed_to_unsigned(x, bw):
    temp = x
    if temp.ndim == 1:
        temp = np.expand_dims(temp, axis=-1)
    # print("Temp Size: ", temp.shape)
    for i in range(0, temp.shape[-1]):
        # print("0")
        idx_below_zero = np.argwhere(temp[:, i] < 0)
        # print("idx_below_zero", idx_below_zero.shape)
        abs_below_zero = np.abs(temp[idx_below_zero, i])
        # print("abs_below_zero", abs_below_zero.shape)
        max_val = 2 ** bw
        unsigned_temp = max_val - abs_below_zero
        temp[idx_below_zero, i] = unsigned_temp
    return temp


def gen_sim_file(file_path, pdict):
    """
    Generate Modelsim Simulation Files for NN parameters

    :param pdict:    Dictionary from which to save numpy arrays of NN parameters in fp
    :param qi:       number of integer bits before the decimal point in fxp
    :param qf:       number of integer bits after the decimal point in fxp
    :return:         mat_param_fxp - parameter in fixed-point numbers
    """

    # Iterate over params in dict
    pdict_out = {}
    for key, value in pdict.items():

        # Get Inputs
        param = value[0]
        qi = value[1]
        qf = value[2]
        idx = value[3]
        idx_bw = value[4]

        if param.ndim == 1:
            param = np.expand_dims(param, axis=-1)
        if idx is not False:
            if idx.ndim == 1:
                idx = np.expand_dims(idx, axis=-1)

        # Get unsigned variable type
        if idx is not False:
            bit_width = qi + qf + idx_bw
        else:
            bit_width = qi + qf

        if 0 < bit_width <= 8:
            var_type = 'uint8'
        elif 8 < bit_width <= 16:
            var_type = 'uint16'
        elif 16 < bit_width <= 32:
            var_type = 'uint32'
        else:
            raise ValueError('Bit width of parameter must not be larger than 32...')

        # Convert fp to fxp
        if idx is not False:
            # param = quantizeTensor(param, qi, qf, 1) # Remove if necessary
            # fxp_param = param
            fxp_param = cast_fp_to_fxp(param, qi, qf)
            fxp_param_unsigned = cast_signed_to_unsigned(fxp_param, qi + qf)
            fxp_param_shift = fxp_param_unsigned * (2 ** idx_bw)
            param = fxp_param_shift + idx
        else:
            param = cast_fp_to_fxp(param, qi, qf)

        # Get param dimension
        dim0 = param.shape[0]
        dim1 = param.shape[1]

        # Create file
        try:
            f = open(file_path + '/' + key + '.txt', 'w')
        except IOError as err:
            print("I/O error({0}): {1}".format(err.errno, err.strerror))

        # Write parameters column-wisely
        param_hex = cast_fxp_to_hex(param, var_type)
        for i in range(0, dim1):
            for j in range(0, dim0):
                if var_type == 'uint8':
                    f.write('%2.2X\n' % param_hex[j][i])
                elif var_type == 'uint16':
                    f.write('%4.4X\n' % param_hex[j][i])
                elif var_type == 'uint32':
                    f.write('%8.8X\n' % param_hex[j][i])
                else:
                    f.write('%X\n' % param_hex[j][i])
        f.close()


def gen_clib(file_path, file_name, pdict, num_layer=None, input_size=None, hid_size=None):
    """
    Generate C Library for NN parameters

    :param filename: name of the C Library
    :param pdict:    Dictionary from which to save numpy arrays of NN parameters in fp
    :param qdict:    Dictionary from which to save properties of corresponding variables in pdict in tuples
        Elements of each tuple in qdict in order:
        - quantize
        - qi:        number of integer bits before the decimal point in fxp
        - qf:        number of integer bits after the decimal point in fxp
        - var_class: 0 - Constant | 1 - General Tensors | 2 - NN Parameter |
        - print_row_size
    :return:         mat_param_fxp - parameter in fixed-point numbers
    """
    print_row_size = 20
    # Create file
    try:
        f_h = open(file_path + '/' + file_name + '.h', 'w')
    except IOError as err:
        print("I/O error({0}): {1}".format(err.errno, err.strerror))
    try:
        f = open(file_path + '/' + file_name + '.c', 'w')
    except IOError as err:
        print("I/O error({0}): {1}".format(err.errno, err.strerror))

    # Write macro definitions
    macro_name = file_name.upper() + '_H'
    f_h.write('#ifndef ' + macro_name + '\n' + '#define ' + macro_name + '\n')
    f.write('#include "' + file_name + '.h"\n')

    # Iterate over params in dict
    for key, value in pdict.items():
        # Get Inputs
        param = np.asarray(value[0])
        qi = value[1]
        qf = value[2]
        idx = value[3]
        idx_bw = value[4]
        param_type = value[5]

        # Get unsigned variable type
        if idx is not False:
            bit_width = qi + qf + idx_bw
        else:
            bit_width = qi + qf

        if 0 < bit_width <= 8:
            var_type = 'char'
            nn_var_type = 'unsigned char'
            np_var_type = 'uint8'
        elif 8 < bit_width <= 16:
            var_type = 'short'
            nn_var_type = 'unsigned short'
            np_var_type = 'uint16'
        elif 16 < bit_width <= 32:
            var_type = 'long'
            nn_var_type = 'unsigned long'
            np_var_type = 'uint32'
        else:
            raise ValueError('Bit width of parameter must not be larger than 32...')

        # Convert fp to fxp
        if param_type != 0:
            if idx is not False:
                # param = quantizeTensor(param, qi, qf, 1) # Remove if necessary
                fxp_param = cast_fp_to_fxp(param, qi, qf)
                fxp_param_unsigned = cast_signed_to_unsigned(fxp_param, qi + qf)
                fxp_param_shift = fxp_param_unsigned * (2 ** idx_bw)
                param = fxp_param_shift + idx
                param = cast_fxp_to_hex(param, np_var_type)
                param = param
            else:
                param = cast_fp_to_fxp(param, qi, qf)

        if param_type == 0:  # Constant
            f_h.write('#define ' + key.upper() + ' %d\n' % param)
        elif param_type == 1:  # General Tensor
            if param.ndim == 1:
                param = np.expand_dims(param, axis=0)

            # Get param dimension
            size = param.size
            dim0 = param.shape[0]
            dim1 = param.shape[1]

            # Write variable declaration
            f_h.write('/*\n'
                      ' * Var Type: General Matrix \n'
                      ' * Var Name: %s' % key + '\n'
                                                ' * Bit Width: %d' % bit_width + '\n'
                                                                                 ' * Dimension: (%d, %d)' % (
                          dim0, dim1) + '\n'
                                        ' */\n'
                      )
            f_h.write('#define ' + key.upper() + '_NUM_ROWS %d\n' % dim0)
            f_h.write('#define ' + key.upper() + '_NUM_COLS %d\n' % dim1)
            f_h.write('#define ' + key.upper() + '_MAT_SIZE %d\n' % size)
            f_h.write('extern const %s %s[' % (var_type, key) + key.upper() + '_MAT_SIZE];\n')
            f.write('const %s %s[' % (var_type, key) + key.upper() + '_MAT_SIZE] = {\n')

            # Write parameters column-wisely
            for i in range(0, dim1):
                for j in range(0, dim0):
                    if var_type == 'float':
                        f.write('%.20f' % param[j][i])
                    else:
                        f.write('%d' % param[j][i])
                    if not (i >= dim1 - 1 and j >= dim0 - 1):
                        f.write(',')
                    if (i * dim0 + j) % print_row_size == print_row_size - 1:
                        f.write('\n')
            f.write('};\n\n')

        elif param_type == 2:  # NN Parameter
            if param.ndim == 1:
                param = np.expand_dims(param, axis=0)

            # Get param dimension
            size = param.size
            dim0 = param.shape[0]
            dim1 = param.shape[1]

            # Write variable declaration
            f_h.write('/*\n'
                      ' * Var Type: NN Parameter Matrix \n'
                      ' * Var Name: %s' % key + '\n'
                                                ' * Bit Width: %d' % bit_width + '\n'
                                                                                 ' * Dimension: (%d, %d)' % (
                          dim0, dim1) + '\n'
                                        ' */\n'
                      )

            f_h.write('#define ' + key.upper() + '_NUM_ROWS %d\n' % dim0)
            f_h.write('#define ' + key.upper() + '_NUM_COLS %d\n' % dim1)
            f_h.write('#define ' + key.upper() + '_MAT_SIZE %d\n' % size)
            f_h.write('extern const %s %s[' % (
                nn_var_type, key) + key.upper() + '_MAT_SIZE] __attribute__ ((aligned (%d)));\n' % bit_width)
            f.write('const %s %s[' % (nn_var_type, key) + key.upper() + '_MAT_SIZE] = {\n')
            if num_layer is not None:
                f_h.write('#define ' + key.upper() + '_NUM_LAYERS %d\n' % num_layer)
            if input_size is not None:
                f_h.write('#define ' + key.upper() + '_INP_SIZE %d\n' % input_size)
            if hid_size is not None:
                f_h.write('#define ' + key.upper() + '_HID_SIZE %d\n' % hid_size)
            # if 'rnn' in key:
            #     f_h.write('extern const %s %s[RNN_PARAM_Hweight_tensor_SIZE] __attribute__ ((aligned (%d)));\n' % (var_type, key, bit_width))
            #     f.write('const %s %s[RNN_PARAM_Hweight_tensor_SIZE] = {\n' % (var_type, key))
            # else:
            #     f_h.write('extern const %s %s[%d];\n' % (var_type, key, size))
            #     f.write('const %s %s[%d] = {\n' % (var_type, key, size))

            # Write parameters column-wisely
            for i in range(0, dim1):
                for j in range(0, dim0):
                    f.write('%d' % param[j][i])
                    if not (i >= dim1 - 1 and j >= dim0 - 1):
                        f.write(',')
                    if (i + j) % print_row_size == print_row_size - 1:
                        f.write('\n')
            f.write('};\n\n')

    f_h.write('#endif')
    f.close()
    f_h.close()

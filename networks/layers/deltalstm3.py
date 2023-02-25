import numpy as np
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
# from networks.nn_util import quantize_tensor, hardsigmoid
from typing import List



# TODO: +hardsigmoid, hardtanh options



################################################################
# Utility functions
# Quantize a tensor without preserving gradients
def quantize_tensor(x: Tensor, qi: int, qf: int, q_type: int = 0):
    """
    :param x: input tensor
    :param qi: number of integer bits before the decimal point
    :param qf: number of fraction bits after the decimal point
    :param q_type: Type of Quantziation (0 - none, 1 - round, 2 - floor)
    :param use_floor: Whether use floor() instead of round()
    :return: tensor quantized to fixed-point precision
    """
    if q_type == 0:
        return x
    else:
        power = float(2. ** qf)
        clip_val = float(2. ** (qi + qf - 1))
        if q_type == 2:
            # value = GradPreserveFloor.apply(x * power)    # Not supported in jit
            value = t.floor(x * power)
        else:
            # value = GradPreserveRound.apply(x * power)    # Not supported in jit
            value = t.round(x * power)
        value = t.clamp(value, -clip_val, clip_val - 1)  # saturation arithmetic
        value = value / power
        return value



################################################################
# Custom LSTM function (jit) - forward 1 timestep
@t.jit.script
def DeltaLSTM_forward_step(xh_curr: Tensor, xh_hat_prev: Tensor,
                           mems_prev: Tensor, c_prev: Tensor,
                           weights: Tensor,
                           th_xh: Tensor, qpl: List[int]):
    qa, aqi, aqf, nqi, nqf, _, _, _ = qpl
    
    xh_dif_curr = xh_curr - xh_hat_prev
    xh_dif_curr_abs = t.abs(xh_dif_curr)
    xh_msk_curr = xh_dif_curr_abs < th_xh
    xh_hat_curr = t.where(xh_msk_curr, xh_hat_prev, xh_curr)            # TODO: add to args
    # xh_hat_curr = xh_hat_prev * xh_msk_curr + xh_curr * (~xh_msk_curr)  # TODO: add to args
    xh_del_curr = xh_hat_curr - xh_hat_prev

    mems_curr = t.addmm(mems_prev, xh_del_curr, weights.t())                                        # TODO: add to args
    # mems_curr = t.addmm(mems_prev.float(), xh_del_curr.float(), weights.t().float()).to(t.bfloat16) # TODO: add to args
    mems_curr = quantize_tensor(mems_curr, aqi, aqf, qa)
    mems_curr_chunks = mems_curr.chunk(4, dim=1)

    acc_i_curr = mems_curr_chunks[0]
    acc_f_curr = mems_curr_chunks[1]
    acc_g_curr = mems_curr_chunks[2]
    acc_o_curr = mems_curr_chunks[3]

    i_curr = t.sigmoid(acc_i_curr)
    f_curr = t.sigmoid(acc_f_curr)
    g_curr = t.tanh(acc_g_curr)
    o_curr = t.sigmoid(acc_o_curr)

    gates_curr = t.cat([i_curr, f_curr, g_curr, o_curr], dim=1)
    gates_curr = quantize_tensor(gates_curr, nqi, nqf, qa)      # aqi, aqf
    i_curr, f_curr, g_curr, o_curr = gates_curr.chunk(4, dim=1)

    c_curr = t.mul(f_curr, c_prev) + t.mul(i_curr, g_curr)
    c_curr = quantize_tensor(c_curr, aqi, aqf, qa)
    c_tanh_curr = t.tanh(c_curr)
    c_tanh_curr = quantize_tensor(c_tanh_curr, nqi, nqf, qa)    # aqi, aqf

    h_curr = t.mul(o_curr, c_tanh_curr)
    h_curr = quantize_tensor(h_curr, aqi, aqf, qa)

    return xh_msk_curr, xh_hat_curr, xh_del_curr, mems_curr, h_curr, c_curr, gates_curr, c_tanh_curr

################################################################
# Custom LSTM function (jit) - forward
@t.jit.script
def DeltaLSTM_forward(input: Tensor, weights: Tensor,
                      x_p_0: Tensor, h_0: Tensor, h_p_0: Tensor, c_0: Tensor, dm_0: Tensor,
                      th_x: float, th_h: float, qpl: List[int]):
    # input (seq_len, batch, n_feat)
    Nt, Nb, Nx = input.size()
    Nh = h_0.size()[1]
    device = input.device
    qa, aqi, aqf, nqi, nqf, qg, gqi, gqf = qpl

    input = quantize_tensor(input, aqi, aqf, qa)

    # Regularizer
    reg = t.zeros(1, dtype=input.dtype, device=input.device).squeeze()

    h_curr = quantize_tensor(h_0, aqi, aqf, qa)
    c_curr = quantize_tensor(c_0, aqi, aqf, qa)
    xh_hat_curr = t.cat([x_p_0, h_p_0], dim=1)
    mems_curr = dm_0

    th_xh = t.cat([t.full((Nb, Nx), th_x, device=device),
                   t.full((Nb, Nh), th_h, device=device)], dim=1)

    output = []
    xh_msk = []
    xh_del = []
    gates = []
    c = []
    c_tanh = []

    for x_curr in input:
        xh_hat_prev = xh_hat_curr
        h_prev = h_curr
        c_prev = c_curr
        mems_prev = mems_curr

        xh_curr = t.cat([x_curr, h_prev], dim=1)
        xh_msk_curr, xh_hat_curr, xh_del_curr, mems_curr, h_curr, c_curr, gates_curr, c_tanh_curr = \
            DeltaLSTM_forward_step(xh_curr, xh_hat_prev, mems_prev, c_prev, weights, th_xh, qpl)
    
        output.append(h_curr)
        xh_msk.append(xh_msk_curr)
        xh_del.append(xh_del_curr)
        gates.append(gates_curr)
        c.append(c_curr)
        c_tanh.append(c_tanh_curr)

    c.append(c_0)

    output = t.stack(output, dim=0)
    xh_msk = t.stack(xh_msk, dim=0)
    xh_del = t.stack(xh_del, dim=0)
    gates = t.stack(gates, dim=0)
    c = t.stack(c, dim=0)
    c_tanh = t.stack(c_tanh, dim=0)

    h_n = h_curr    # output[-1, :, :].view(1, output.size(1), output.size(2))
    c_n = c_curr    # c[-1, :, :].view(1, c.size(1), c.size(2))
    x_hat_n, h_hat_n = xh_hat_curr.split([Nx, Nh], dim=1)
    mems_n = mems_curr

    x_del, h_del = xh_del.split([Nx, Nh], dim=2)
    reg = t.sum(t.abs(h_del))               # TODO: add to args
    # reg = t.sum(t.abs(h_del.float()))       # TODO: add to args, FP16
    # sp_dx = t.tensor(float(t.numel(x_del) - t.nonzero(x_del).size(0)) / t.numel(x_del))
    # sp_dh = t.tensor(float(t.numel(h_del) - t.nonzero(h_del).size(0)) / t.numel(h_del))
    dbl = [t.numel(x_del) - t.nonzero(x_del).size(0),   # num_dx_zeros
           t.numel(h_del) - t.nonzero(h_del).size(0),   # num_dh_zeros
           t.numel(x_del),                              # num_dx_numel
           t.numel(h_del)]                              # num_dh_numel
    
    return output, (x_hat_n, h_n, h_hat_n, c_n, mems_n), reg, dbl, \
           xh_msk, xh_del, gates, c, c_tanh



################################################################
# Custom LSTM function (jit) - backward preprocessing
@t.jit.script
def DeltaLSTM_backward_preproc(gates: Tensor, c: Tensor, c_tanh: Tensor,
                               gqi: int, gqf: int, qg: int):
    i, f, g, o = gates.chunk(4, dim=2)
    c_p1 = t.cat([c[-1:, :, :], c[:-2, :, :]], dim=0)

    dtanhc_dc = t.add(1, -t.mul(c_tanh, c_tanh))
    dtanhc_dc = quantize_tensor(dtanhc_dc, gqi, gqf, qg)

    dh_dc = o * dtanhc_dc
    dh_dc = quantize_tensor(dh_dc, gqi, gqf, qg)

    di_dZi = t.mul(i, t.add(1, -i))
    df_dZf = t.mul(f, t.add(1, -f))
    dg_dZg = t.add(1, -t.mul(g, g))
    do_dZo = t.mul(o, t.add(1, -o))
    dG_dZ = t.cat([di_dZi, df_dZf, dg_dZg, do_dZo], dim=2)
    dG_dZ = quantize_tensor(dG_dZ, gqi, gqf, qg)
    
    dch_dG = t.cat([g, c_p1, i, c_tanh], dim=2)
    dch_dZ = dch_dG * dG_dZ
    dch_dZ = quantize_tensor(dch_dZ, gqi, gqf, qg)

    return dh_dc, dch_dZ, f

################################################################
# Custom LSTM function (jit) - backward 1 timestep
@t.jit.script
def DeltaLSTM_backward_step(grad_output: Tensor, dL_dh_curr: Tensor, dL_dc_curr: Tensor, dL_dZ_curr: Tensor,
                            dh_dc_curr: Tensor, dch_dZ_curr: Tensor, f_curr: Tensor,
                            grad_weights: Tensor, weights: Tensor,
                            dL_dhxh_curr: Tensor, xh_msk_curr: Tensor, xh_del_curr: Tensor, xh_del_reg_msk_curr: Tensor, grad_reg: Tensor,
                            Nx: int, gqi: int, gqf: int, qg: int):

    dL_dh_curr += grad_output
    dL_dh_curr = quantize_tensor(dL_dh_curr, gqi, gqf, qg)

    dL_dc_curr = dL_dc_curr + dL_dh_curr * dh_dc_curr
    dL_dc_curr = quantize_tensor(dL_dc_curr, gqi, gqf, qg)
    
    dL_dch_curr = t.cat([dL_dc_curr, dL_dc_curr, dL_dc_curr, dL_dh_curr], dim=1)
    dL_dZ_curr += dL_dch_curr * dch_dZ_curr
    dL_dZ_curr = quantize_tensor(dL_dZ_curr, gqi, gqf, qg)

    dL_dc_prev = f_curr * dL_dc_curr
    dL_dc_prev = quantize_tensor(dL_dc_prev, gqi, gqf, qg)

    dL_dZ_prev = dL_dZ_curr

    grad_weights += t.matmul(dL_dZ_curr.t(), xh_del_curr)   # TODO: add to args
    dL_ddxh_curr = t.matmul(dL_dZ_curr, weights)            # TODO: add to args
    # grad_weights += t.matmul(dL_dZ_curr.t().float(), xh_del_curr.float()).to(t.bfloat16)    # TODO: add to args
    # dL_ddxh_curr = t.matmul(dL_dZ_curr.float(), weights.float()).to(t.bfloat16)             # TODO: add to args

    xh_del_reg_curr = t.sign(xh_del_curr) * grad_reg * xh_del_reg_msk_curr    # TODO: add to args
    # xh_del_reg_curr = t.zeros_like(dL_ddxh_curr)                                            # TODO: add to args
    # xh_del_reg_curr.masked_fill_(xh_del_curr > 0, grad_reg)
    # xh_del_reg_curr.masked_fill_(xh_del_curr == 0, 0)
    # xh_del_reg_curr.masked_fill_(xh_del_curr < 0, -grad_reg)
    
    xh_del_reg_curr *= xh_del_reg_msk_curr
    dL_ddxh_curr = dL_ddxh_curr + xh_del_reg_curr

    dL_dxh_curr = (dL_dhxh_curr + dL_ddxh_curr) * (~xh_msk_curr)
    dL_dhxh_prev = dL_dhxh_curr * xh_msk_curr - dL_ddxh_curr * (~xh_msk_curr)
    dL_dxh_curr = quantize_tensor(dL_dxh_curr, gqi, gqf, qg)
    dL_dhxh_prev = quantize_tensor(dL_dhxh_prev, gqi, gqf, qg)

    return dL_dh_curr, dL_dc_prev, dL_dZ_prev, grad_weights, dL_dxh_curr, dL_dhxh_prev

################################################################
# Custom LSTM function (jit) - backward
@t.jit.script
def DeltaLSTM_backward(grad_output: Tensor, grad_states: Tensor, grad_reg: Tensor,
                       weights: Tensor, xh_msk: Tensor, xh_del: Tensor, gates: Tensor, c: Tensor, c_tanh: Tensor,
                       qpl:List[int]):
    # x (seq_len, batch, n_feat)
    Nt, Nb, Nh = c.size()
    Nt = Nt - 1
    Nx = xh_del.size()[2] - Nh
    device = weights.device
    _, _, _, _, _, qg, gqi, gqf = qpl

    grad_output = quantize_tensor(grad_output, gqi, gqf, qg)

    grad_input = []
    # grad_input   = t.zeros(Nt, Nb, Nx, device=device)
    grad_weights = t.zeros(Nh*4, Nx+Nh, device=device)

    dh_dc, dch_dZ, f = DeltaLSTM_backward_preproc(gates, c, c_tanh, gqi, gqf, qg)

    dL_dh_curr = t.zeros(Nb, Nh, device=device)
    dL_dc_curr = t.zeros(Nb, Nh, device=device)
    dL_dZ_curr = t.zeros(Nb, Nh*4, device=device)
    dL_dhxh_curr = t.zeros(Nb, Nx+Nh, device=device)
    xh_del_reg_msk_curr = t.cat([t.zeros(Nb, Nx, device=device),
                                 t.ones(Nb, Nh, device=device)], dim=1)

    # for ti in range(Nt-1, -1, -1):                  # TODO: add to args
    #     dL_dh_curr, dL_dc_curr, dL_dZ_curr, grad_weights, dL_dxh_curr, dL_dhxh_curr = DeltaLSTM_backward_step(grad_output[ti], dL_dh_curr, dL_dc_curr, dL_dZ_curr,
    #                 dh_dc[ti], dch_dZ[ti], f[ti],
    #                 grad_weights, weights,
    #                 dL_dhxh_curr, xh_msk[ti], xh_del[ti], xh_del_reg_msk_curr, grad_reg,
    #                 Nx, gqi, gqf, qg)
    #     grad_input[ti], dL_dh_curr = dL_dxh_curr.split([Nx, Nh], dim=1)
    for grad_output_curr, dh_dc_curr, dch_dZ_curr, f_curr, xh_msk_curr, xh_del_curr \
        in zip(grad_output.flip(dims=[0]),          # TODO: add to args
               dh_dc.flip(dims=[0]),
               dch_dZ.flip(dims=[0]),
               f.flip(dims=[0]),
               xh_msk.flip(dims=[0]),
               xh_del.flip(dims=[0])):
        dL_dh_curr, dL_dc_curr, dL_dZ_curr, grad_weights, dL_dxh_curr, dL_dhxh_curr = \
            DeltaLSTM_backward_step(grad_output_curr, dL_dh_curr, dL_dc_curr, dL_dZ_curr,
                                    dh_dc_curr, dch_dZ_curr, f_curr,
                                    grad_weights, weights,
                                    dL_dhxh_curr, xh_msk_curr, xh_del_curr, xh_del_reg_msk_curr, grad_reg,
                                    Nx, gqi, gqf, qg)
        grad_input_curr, dL_dh_curr = dL_dxh_curr.split([Nx, Nh], dim=1)
        grad_input.append(grad_input_curr)
    
    grad_input.reverse()
    grad_input = t.stack(grad_input, dim=0)

    grad_mems_0 = dL_dZ_curr
    
    grad_input   = quantize_tensor(grad_input  , gqi, gqf, qg)
    grad_weights = quantize_tensor(grad_weights, gqi, gqf, qg)
    grad_mems_0  = quantize_tensor(grad_mems_0, gqi, gqf, qg)

    return grad_input, grad_weights, grad_mems_0



################################################################
# Custom LSTM function
#   One layer
#   Custom forward and custom backward
class DeltaLSTMFunction(t.autograd.function.Function):

    @staticmethod
    def forward(ctx, input: Tensor, weights: Tensor,
                x_p_0: Tensor, h_0: Tensor, h_p_0: Tensor, c_0: Tensor, dm_0: Tensor,
                th_x: float, th_h: float, qpl: List[int]):
        
        output, (x_p_n, h_n, h_p_n, c_n, dm_n), reg, dbl, \
            xh_msk, xh_del, gates, c, c_tanh = \
            DeltaLSTM_forward(input, weights, x_p_0, h_0, h_p_0, c_0, dm_0,
                              th_x, th_h, qpl)

        ctx.save_for_backward(weights, xh_msk, xh_del, gates, c, c_tanh)
        ctx.qpl = qpl

        return output, (x_p_n, h_n, h_p_n, c_n, dm_n), reg, dbl

    @staticmethod
    # Inputs are the gradients of the outputs of the forward() function
    def backward(ctx, grad_output: Tensor, grad_states: Tensor, grad_reg: Tensor, grad_dbl: Tensor):
        # grad_x_p_out, grad_h, grad_h_p, grad_c, grad_dm = grad_states
        weights, xh_msk, xh_del, gates, c, c_tanh = ctx.saved_tensors
        qpl = ctx.qpl

        grad_input, grad_weights, grad_dm_0 = \
            DeltaLSTM_backward(grad_output, grad_states, grad_reg,
                                weights, xh_msk, xh_del, gates, c, c_tanh,
                                qpl)

        if not ctx.needs_input_grad[0]:
            grad_input = None
        
        # Return gradients of the inputs of the forward() function
        return grad_input, grad_weights, \
               None, None, None, None, grad_dm_0, \
               None, None, None, None



################################################################
# Custom LSTM module
class DeltaLSTM3(nn.LSTM):
    def __init__(self,
                 input_size=16,
                 hidden_size=256,
                 num_layers=2,
                #  dropout=0,   # TODO: add dropout
                 thx=0,
                 thh=0,
                 qa=0,
                 aqi=8,
                 aqf=8,
                 qw=0,
                 wqi=1,
                 wqf=7,
                 nqi=2,
                 nqf=4,
                 qg=0,      # TODO: add qg
                 gqi=8,
                 gqf=8,
                 bw_acc=32,
                 use_hardsigmoid=0,
                 use_hardtanh=0,
                 debug=0):
        super(DeltaLSTM3, self).__init__(input_size, hidden_size, num_layers)

        # Hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.th_x = thx
        self.th_h = thh
        # self.dropout = dropout
        self.qa = qa
        self.aqi = aqi
        self.aqf = aqf
        self.qw = qw
        self.wqi = wqi
        self.wqf = wqf
        self.nqi = nqi
        self.nqf = nqf
        self.qg = qg
        self.gqi = gqi
        self.gqf = gqf
        self.bw_acc = bw_acc
        self.debug = debug
        self.weight_ih_height = 4 * self.hidden_size  # Wih has 4 weight matrices stacked vertically
        self.weight_ih_width = self.input_size
        self.weight_hh_width = self.hidden_size
        self.use_hardsigmoid = use_hardsigmoid
        self.use_hardtanh = use_hardtanh
        self.x_p_length = max(self.input_size, self.hidden_size)

        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Debug
        self.set_debug(self.debug)

    def set_debug(self, value):
        setattr(self, "debug", value)
        self.dict_debug = {
            "num_dx_zeros": 0,
            "num_dx_numel": 0,
            "num_dh_zeros": 0,
            "num_dh_numel": 0
        }

    def add_to_debug(self, x, i_layer, name):
        if self.debug:
            if isinstance(x, Tensor):
                variable = np.squeeze(x.cpu().numpy())
            else:
                variable = np.squeeze(np.asarray(x))
            variable_name = '_'.join(['l' + str(i_layer), name])
            if variable_name not in self.dict_debug.keys():
                self.dict_debug[variable_name] = []
            self.dict_debug[variable_name].append(variable)

    def init_weight(self):
        for name, param in self.named_parameters():
            print('::: Initializing Parameters: ', name)
            if 'l0' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param[:self.hidden_size, :])
                    nn.init.xavier_uniform_(param[self.hidden_size:2 * self.hidden_size, :])
                    nn.init.xavier_uniform_(param[2 * self.hidden_size:3 * self.hidden_size, :])
                    nn.init.xavier_uniform_(param[3 * self.hidden_size:, :])
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param[:self.hidden_size, :])
                    nn.init.orthogonal_(param[self.hidden_size:2 * self.hidden_size, :])
                    nn.init.orthogonal_(param[2 * self.hidden_size:3 * self.hidden_size, :])
                    nn.init.orthogonal_(param[3 * self.hidden_size:, :])
            else:
                if 'weight' in name:
                    nn.init.orthogonal_(param[:self.hidden_size, :])
                    nn.init.orthogonal_(param[self.hidden_size:2 * self.hidden_size, :])
                    nn.init.orthogonal_(param[2 * self.hidden_size:3 * self.hidden_size, :])
                    nn.init.orthogonal_(param[3 * self.hidden_size:, :])
            if 'bias' in name:
                nn.init.constant_(param, 0)
        print("--------------------------------------------------------------------")

    def process_inputs(self, x: Tensor, qa: int, x_p_0: Tensor = None, h_0: Tensor = None, h_p_0: Tensor = None,
                       c_0: Tensor = None, dm_0: Tensor = None):
        """
        Process DeltaGRU Inputs (please refer to the DeltaGRU formulations)
        :param x:     x(t), Input Tensor
        :param x_p_0: x(t-1), Input Tensor
        :param h_0:   h(t-1), Hidden state
        :param h_p_0: h(t-2), Hidden state
        :param c_0:   c(t-1), Cell State
        :param dm_0:  dm(t-1), Delta Memory
        :return: initialized state tensors
        """

        # Reshape input if necessary
        if self.batch_first:
            x.transpose(0, 1)
            setattr(self, 'batch_size', int(x.size()[0]))
        else:
            setattr(self, 'batch_size', int(x.size()[1]))
        batch_size = x.size()[1]
        x = quantize_tensor(x, self.aqi, self.aqf, qa)

        if x_p_0 is None or h_0 is None or h_p_0 is None or c_0 is None or dm_0 is None:
            # Generate zero state if external state not provided
            x_p_0 = t.zeros(self.num_layers, batch_size, self.x_p_length,
                                dtype=x.dtype, device=x.device)
            h_0 = t.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            h_p_0 = t.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            c_0 = t.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            dm_0 = t.zeros(self.num_layers, batch_size, self.weight_ih_height, dtype=x.dtype, device=x.device)
            for l in range(self.num_layers):
                bias_ih = getattr(self, 'bias_ih_l{}'.format(l))
                bias_hh = getattr(self, 'bias_hh_l{}'.format(l))
                dm_0[l, ...] = quantize_tensor(dm_0[l, ...] + bias_ih + bias_hh, self.wqi, self.wqf, self.qw)

        return x, x_p_0, h_0, h_p_0, c_0, dm_0

    def forward(self, input: Tensor, x_p_0: Tensor = None, h_0: Tensor = None, h_p_0: Tensor = None,
                c_0: Tensor = None, dm_0: Tensor = None):
        # Quantize
        qa = 0 if self.training else self.qa
        # qa = self.qa
        
        # Quantization parameter list
        qpl = [qa, self.aqi, self.aqf, self.nqi, self.nqf, self.qg, self.gqi, self.gqf]

        # Initialize State
        x, x_p_0, h_0, h_p_0, c_0, dm_0 = self.process_inputs(input, qa, x_p_0, h_0, h_p_0, c_0, dm_0)

        # Quantize threshold
        th_x = quantize_tensor(t.tensor(self.th_x, dtype=input.dtype), self.aqi, self.aqf, qa)
        th_h = quantize_tensor(t.tensor(self.th_h, dtype=input.dtype), self.aqi, self.aqf, qa)
        
        # Iterate through layers
        reg = t.zeros(1, dtype=x.dtype, device=input.device).squeeze()
        x_p_n = []
        h_n = []
        h_p_n = []
        c_n = []
        dm_n = []
        
        for l in range(self.num_layers):
            # Concatenate weights
            weight_ih = getattr(self, 'weight_ih_l{}'.format(l))
            weight_hh = getattr(self, 'weight_hh_l{}'.format(l))
            weights = t.cat([weight_ih, weight_hh], dim=1)
            
            # x, (x_p_n_l, h_n_l, h_p_n_l, c_n_l, dm_n_l), reg_l = self.layer_forward(x, l, qpl, x_p_0[l], h_0[l],
            #                                                                       h_p_0[l], c_0[l], dm_0[l])
            func_output = DeltaLSTMFunction.apply(x, weights,
                                                  x_p_0[l, :, :self.input_size], h_0[l], h_p_0[l], c_0[l], dm_0[l],
                                                  th_x, th_h, qpl)
            x, (x_p_n_l, h_n_l, h_p_n_l, c_n_l, dm_n_l), reg_l, dbl = func_output
            
            x_p_n.append(x_p_n_l)
            h_n.append(h_n_l)
            h_p_n.append(h_p_n_l)
            c_n.append(c_n_l)
            dm_n.append(dm_n_l)
            reg += reg_l
            
            if self.debug:
                self.dict_debug["num_dx_zeros"] += dbl[0]
                self.dict_debug["num_dh_zeros"] += dbl[1]
                self.dict_debug["num_dx_numel"] += dbl[2]
                self.dict_debug["num_dh_numel"] += dbl[3]
        
        x_p_n = t.stack(x_p_n)
        h_n = t.stack(h_n)
        h_p_n = t.stack(h_p_n)
        c_n = t.stack(c_n)
        dm_n = t.stack(dm_n)


        # Debug
        if self.debug:
            self.dict_debug["sparsity_dx"] = float(self.dict_debug["num_dx_zeros"] / self.dict_debug["num_dx_numel"])
            self.dict_debug["sparsity_dh"] = float(self.dict_debug["num_dh_zeros"] / self.dict_debug["num_dh_numel"])
            self.dict_debug["sparsity_to"] = float((self.dict_debug["num_dx_zeros"] + self.dict_debug["num_dh_zeros"]) /
                                                   (self.dict_debug["num_dx_numel"] + self.dict_debug["num_dh_numel"]))

        return x, (x_p_n, h_n, h_p_n, c_n, dm_n), reg

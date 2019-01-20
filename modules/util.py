# __author__ = "Chang Gao"
# __copyright__ = "Copyright 2018 to the author"
# __license__ = "Private"
# __version__ = "0.1.0"
# __maintainer__ = "Chang Gao"
# __email__ = "chang.gao@uzh.ch"
# __status__ = "Prototype"

import torch as t
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = t.typename(x).split('.')[-1]
    sparse_tensortype = getattr(t.sparse, x_typename)

    indices = t.nonzero(x)
    if indices.nelement() == 0:  # if all elements are zeros
        print("1", indices)
        return sparse_tensortype(*x.shape)
    else:
        print("2", indices)
        indices = indices.t()
        values = x[tuple(indices[i] for i in range(indices.shape[0]))]
        return sparse_tensortype(indices, values, x.size())

def quantizeTensor(x, m, n):
    """
    :param x: input tensor
    :param m: number of integer bits before the decimal point
    :param n: number of fraction bits after the decimal point
    :return: tensor quantized to fixed-point precision
    """
    power = 2. ** n
    max_val = 2. ** (m + n) - 1
    value = x * power
    value = GradPreserveRoundOp.apply(value)  # rounding
    value = t.clamp(value, -max_val, max_val)  # saturation arithmetic
    value = value / power
    return value

class GradPreserveRoundOp(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input):
        output = t.round(input)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        grad_input = grad_output

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        # print(grad_output.size())
        # if not t.equal(grad_output, QuantizeT(grad_output, dW_qp)): print("grad_output not quantized")
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        # Return same number of parameters as "def forward(...)"
        return grad_input

class GradPreserveThreshold(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, threshold, value):
        output = F.threshold(input, threshold, value)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        grad_input = grad_output

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        # print(grad_output.size())
        # if not t.equal(grad_output, QuantizeT(grad_output, dW_qp)): print("grad_output not quantized")
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        # Return same number of parameters as "def forward(...)"
        return grad_input

def look_ahead_seq(seq_in, t_width=16, padding=0, batch_first=0):

    # Convert input sequence to batch first shape (seq_len, n_batch, n_feature)
    seq = seq_in
    if batch_first:
        seq = seq_in.transpose(0, 1)
    
    seq_len = seq.size(0)
    n_batch = seq.size(1)
    n_feature = seq.size(2)
    #int(t.ceil(float(seq_len)/float(t_width)))
    new_seq = []
    for i in range(0, seq_len):
        if i < seq_len - t_width:
            seq_block = seq[i:i+t_width, :, :]
        else:
            seq_block = seq[i:, :, :]
            seq_block_pad = t.zeros([t_width-(seq_len-i), n_batch, n_feature], dtype=t.float32).cuda()
            seq_block = t.cat((seq_block, seq_block_pad), 0)
        new_seq.append(seq_block)
    new_seq = t.stack(new_seq, 0)
    new_seq = new_seq.transpose(1, 2)
    new_seq = new_seq.transpose(0, 1)
    new_seq = new_seq.transpose(2, 3)
    return new_seq


def look_around_seq(seq_in, t_width=16, padding=0, batch_first=0):
    # Convert input sequence to batch first shape (seq_len, n_batch, n_feature)
    seq = seq_in
    if batch_first:
        seq = seq_in.transpose(0, 1)

    seq_len = seq.size(0)
    n_batch = seq.size(1)
    n_feature = seq.size(2)
    # int(t.ceil(float(seq_len)/float(t_width)))
    new_seq = []
    for i in range(0, seq_len):
        if i >= seq_len - t_width:
            seq_block = seq[i-t_width:, :, :]
            seq_block_pad = t.zeros([t_width - (seq_len - i) + 1, n_batch, n_feature], dtype=t.float32).cuda()
            seq_block = t.cat((seq_block, seq_block_pad), 0)
        elif i < t_width:
            seq_block = seq[0:i + 1 + t_width, :, :]
            seq_block_pad = t.zeros([t_width - i, n_batch, n_feature], dtype=t.float32).cuda()
            seq_block = t.cat((seq_block, seq_block_pad), 0)
        else:
            seq_block = seq[i-t_width:i + 1 + t_width, :, :]
        #print(seq_block.size())
        new_seq.append(seq_block)
    new_seq = t.stack(new_seq, 0)
    new_seq = new_seq.transpose(1, 2)
    new_seq = new_seq.transpose(0, 1)
    new_seq = new_seq.transpose(2, 3)
    return new_seq




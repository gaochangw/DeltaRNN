import torch
from torch import Tensor
from torch.autograd.function import Function
import torch.nn.functional as F

class GradPreserveFloor(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.floor(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        return grad_input


class GradPreserveRound(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        return grad_input


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
            value = GradPreserveFloor.apply(x * power)
        else:
            value = GradPreserveRound.apply(x * power)
        value = torch.clamp(value, -clip_val, clip_val - 1)  # saturation arithmetic
        value = value / power
        return value


def hardsigmoid(x):
    """
    Computes element-wise hard sigmoid of x.
    See e.g. https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
    """
    x = (0.25 * x) + 0.5
    x = F.threshold(-x, -1.0, -1.0)
    x = F.threshold(-x, 0.0, 0.0)
    return x
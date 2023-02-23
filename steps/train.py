import random as rnd
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from project import Project
from utils import pandaslogger, util
from tqdm import tqdm
import importlib

old_repr = torch.Tensor.__repr__


def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)


def main(proj: Project):
    torch.Tensor.__repr__ = tensor_info
    print("::: Step:    ", proj.step)
    proj.select_device()  # Select CPU/GPU
    proj.reproducible()  # Reproducibility
    proj.build_dataloader()  # Build dataloader
    proj.build_structure()  # Build project structure
    proj.build_criterion()  # Loss Function
    net = proj.build_model()  # Build Neural Network
    proj.build_optimizer()  # Build network optimizer
    proj.build_meter()  # Build accuracy meter
    proj.build_logger()  # Build Logger

    # Starting training
    proj.learn()

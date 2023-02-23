import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from project import Project


def slide_window(seq, window_size, window_stride):
    """
    :param seq: 2-D (T, C) tensor that is the real sequence
    :return seq_window: 3-D (T, window_size, C) tensor that is the slide window of a real sequence
    """
    seq_len = seq.shape[0]
    seq_window = torch.cat([seq[t:t + window_size, :] for t in np.arange(0, seq_len - window_size, window_stride)],
                           dim=0)
    return seq_window


# def get_smoothed_seq(seq, window_size, activation='softmax'):
#     """
#     :param seq: 2-D (T, C) tensor that is the real sequence
#     :param window_size: Size of the sliding window
#     :param activation: Activation function that converts logits to scores
#     :return: smoothed_seq: 2-D (T, C) tensor that is the smoothed sequence
#     """
#     if activation == 'log_softmax':
#         post_seq = F.log_softmax(seq, dim=-1)
#     elif activation == 'softmax':
#         post_seq = F.softmax(seq, dim=-1)
#     elif activation == 'sigmoid':
#         post_seq = torch.sigmoid(seq)
#     else:
#         post_seq = seq
#     # post_seq = torch.sigmoid(seq)
#     seq_len = post_seq.shape[0]
#     num_class = post_seq.shape[1]
#     padded_seq = torch.cat((torch.zeros(window_size - 1, num_class), post_seq), dim=0)
#     # windowed_seq: 3-D (T, window_size, C)
#     windowed_seq = torch.stack([padded_seq[t:t + window_size, :] for t in np.arange(0, seq_len)], dim=0)
#     sum_of_window = torch.sum(windowed_seq, dim=1)
#     denominator = torch.ones(seq_len, 1) * window_size
#     denominator[:min(window_size, seq_len), :] = torch.arange(1, min(window_size + 1, seq_len + 1)).view(-1, 1)
#     smoothed_seq = sum_of_window / denominator
#
#     return smoothed_seq
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_smoothed_seq(seq, window_size, activation='softmax'):
    """
    :param seq: 2-D (T, C) tensor that is the real sequence
    :param window_size: Size of the sliding window
    :param activation: Activation function that converts logits to scores
    :return: smoothed_seq: 2-D (T, C) tensor that is the smoothed sequence
    """
    from scipy.special import log_softmax, softmax
    if activation == 'log_softmax':
        post_seq = log_softmax(seq, axis=-1)
    elif activation == 'softmax':
        post_seq = softmax(seq, axis=-1)
    elif activation == 'sigmoid':
        post_seq = sigmoid(seq)
    else:
        post_seq = seq
    # post_seq = torch.sigmoid(seq)
    seq_len = post_seq.shape[0]
    num_class = post_seq.shape[1]
    zero_pad = np.zeros((window_size, num_class))
    padded_seq = np.concatenate((zero_pad, post_seq), axis=0)
    # windowed_seq: 3-D (T, window_size, C)
    windowed_seq = np.stack([padded_seq[t - window_size:t + 1, :] for t in np.arange(window_size, seq_len + window_size)], axis=0)
    smoothed_seq = np.mean(windowed_seq, axis=1)
    return smoothed_seq


def get_score(seq, window_size, log_softmax=False):
    """
    :param seq: 2-D (T, C) tensor that is the real sequence
    :param window_size: Size of the sliding window
    :return: smoothed_seq: 2-D (T, C) tensor that is the smoothed sequence
    """
    if log_softmax:
        post_seq = torch.abs(F.log_softmax(seq, dim=-1))
    else:
        post_seq = F.softmax(seq, dim=-1)

    seq_len = post_seq.shape[0]
    num_class = post_seq.shape[1]
    padded_seq = torch.cat((torch.zeros(window_size - 1, num_class), post_seq), dim=0)
    # windowed_seq: 3-D (T, window_size, C)
    windowed_seq = torch.stack([padded_seq[t:t + window_size, :] for t in np.arange(0, seq_len)], dim=0)
    # prod_of_window: 2-D (T, C)
    prod_of_window = torch.prod(windowed_seq, dim=1)
    score = np.power(prod_of_window, 1.0 / float(window_size))
    score = 1 / score

    return score


def get_confidence_seq(seq, window_size):
    """
    :param seq: 2-D (T, C) tensor that is the real sequence
    :param window_size: Size of the sliding window
    :return: smoothed_seq: 2-D (T, C) tensor that is the smoothed sequence
    """
    seq_len = seq.shape[0]
    num_kw = seq.shape[1] - 1
    seq_kw = seq[:, 1:]
    padded_seq_kw = torch.cat((torch.zeros(window_size - 1, num_kw), seq_kw), dim=0)
    # windowed_seq: 3-D (T, window_size, C)
    window_seq = torch.stack([padded_seq_kw[t:t + window_size, :] for t in np.arange(0, seq_len)], dim=0)
    window_seq = window_seq.numpy().astype(np.float64)
    # max_prob_window_seq: 2-D (T, C)
    max_prob_window_seq = np.amax(window_seq, axis=1)
    # max_prob_window_seq = max_prob_window_seq.numpy()
    # prod_of_window: 1-D (T)
    prod_of_window = np.prod(max_prob_window_seq, axis=-1)
    confidence_seq = np.power(prod_of_window, 1.0 / float(num_kw))
    confidence_seq = torch.from_numpy(confidence_seq)
    return confidence_seq


def greedy_decoder(post_seq):
    """
    Greedy decoder with threshold
    :param post_seq: 2-D (T, C) tensor that is a posterior sequence
    :return: seq_decoded: 1-D (T) tensor that is the decoded sequence
    """

    idx_pred = np.argmax(post_seq, axis=-1)
    score_pred = np.max(post_seq, axis=-1)
    return score_pred, idx_pred


class Meter:
    def __init__(self, proj: Project):
        self.data = {}

        # Parameters
        self.num_classes = proj.num_classes
        self.idx_silence = 0
        self.smooth = proj.smooth
        self.smooth_window_size = proj.smooth_window_size
        self.confidence_window_size = proj.confidence_window_size
        self.zero_padding = proj.zero_padding
        self.threshold = proj.fire_threshold

    def set_smooth_window_size(self, smooth_window_size):
        self.smooth_window_size = smooth_window_size

    def set_confidence_window_size(self, confidence_window_size):
        self.confidence_window_size = confidence_window_size

    def set_zero_padding(self, zero_padding):
        self.zero_padding = zero_padding

    def add_data(self, **kwargs):
        # Initialize
        for k in kwargs.keys():
            if k not in self.data.keys():
                self.data[k] = []
        # Add data
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                temp = v.detach().cpu().numpy()
            else:
                temp = v
            self.data[k].append(temp)

    def clear_data(self):
        # Clear Data Buffers
        self.data = {}

    def get_real_sequence(self):
        real_seq = []
        real_seq_flag = []
        for i, batch in enumerate(self.outputs):
            for j, seq in enumerate(batch):
                if self.zero_padding == 'tail':
                    real_seq.append(seq[:self.feature_lengths[i][j], :])
                else:
                    real_seq.append(seq[-self.feature_lengths[i][j]:, :])
                real_seq_flag.append(self.flags[i][j].repeat(self.feature_lengths[i][j]))
        real_seq = torch.cat(real_seq, dim=0).float().numpy()
        real_seq_flag = torch.cat(real_seq_flag, dim=0).long().numpy()
        return real_seq, real_seq_flag

    def get_decisions(self, proj):
        for k, v in self.data.items():
            self.data[k] = np.concatenate(v, axis=0)
        # y_true = np.amax(self.data['targets_metric'], axis=-1)
        # Greedy Decoder
        idx_pred = np.argmax(self.data['outputs'], axis=-1)
        score_pred = np.amax(self.data['outputs'], axis=-1)
        seq_len = idx_pred.shape[-1]
        y_pred = []
        for seq in idx_pred:
            nonzero_idx = np.nonzero(seq)[0]
            if nonzero_idx.size != 0:
                decision_idx = np.amax(nonzero_idx)
            else:
                decision_idx = seq_len-1
            y_pred.append(seq[decision_idx])
        y_pred = np.asarray(y_pred)
        y_true = self.data['flags']
        return y_true, y_pred

    def get_metrics(self, dict_stat, proj):
        # Get Decision Sequence
        y_true, y_pred = self.get_decisions(proj)

        # Get Confusion Matrix
        cnf_matrix = confusion_matrix(y_true, y_pred, normalize='pred')

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        # Remove the silent label
        # if len(FP) != self.num_classes - 1:
        #     FP = FP[1:]
        # if len(FN) != self.num_classes - 1:
        #     FN = FN[1:]
        # if len(TP) != self.num_classes - 1:
        #     TP = TP[1:]
        # if len(TN) != self.num_classes - 1:
        #     TN = TN[1:]

        # Sensitivity, hit rate, recall, or true positive rate
        dict_stat['tpr'] = TP / (TP + FN)
        # False negative rate
        dict_stat['fnr'] = FN / (TP + FN)
        # print(dict_stat)
        # Specificity or true negative rate
        dict_stat['tnr'] = TN / (TN + FP)
        # Precision or positive predictive value
        dict_stat['ppv'] = TP / (TP + FP)
        # Negative predictive value
        dict_stat['npv'] = TN / (TN + FN)
        # Fall out or false positive rate
        dict_stat['fpr'] = FP / (FP + TN)

        # False discovery rate
        dict_stat['fdr'] = FP / (TP + FP)
        # Overall accuracy
        dict_stat['acc'] = (TP + TN) / (TP + FP + FN + TN)
        # Micro F1 Score
        dict_stat['f1_score_micro'] = f1_score(y_true, y_pred, average='micro')
        dict_stat['cnf_matrix'] = cnf_matrix

        # Clear Data Buffers
        self.outputs = []
        self.feature_lengths = []
        self.targets = []
        self.target_lengths = []
        self.flags = []

        return dict_stat

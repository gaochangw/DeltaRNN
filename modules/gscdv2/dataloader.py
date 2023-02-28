__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

import os

import h5py
import numpy as np

from utils import util
from utils.util import load_h5py_data, quantize_array
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from project import Project


class DataLoader:
    def __init__(self, proj: Project):
        # Get Arguments
        batch_size = proj.batch_size
        batch_size_test = proj.batch_size_eval

        # Create Datasets
        train_set = GSCDDataset(proj, proj.trainfile)
        dev_set = GSCDDataset(proj, proj.devfile)
        test_set = GSCDDataset(proj, proj.testfile)
        self.num_features = test_set.num_features

        # Check Number of Classes
        num_classes = proj.num_classes
        if proj.loss == 'ctc':
            num_classes += 1  # CTC Label Shift
            proj.additem('num_classes', num_classes)

        train_set, train_set_stat = process_train(proj, train_set)
        dev_set, dev_set_stat = process_test(proj, dev_set, train_set_stat)
        test_set, test_set_stat = process_test(proj, test_set, train_set_stat)

        # Define Collate Function
        def collate_fn(data):
            """
               data: is a list of tuples with (example, label, length)
                     where 'example' is a tensor of arbitrary shape
                     and label/length are scalars
            """
            feature, feature_length, target, target_length, flag = zip(*data)
            feature_lengths = torch.stack(feature_length).long()
            target_lengths = torch.stack(target_length).long()
            flags = torch.stack(flag).long()
            features = pad_sequence(feature)
            targets = pad_sequence(target, batch_first=True)
            return features, feature_lengths, targets, target_lengths, flags

        # Create PyTorch dataloaders for train and dev set
        num_workers = int(proj.num_cpu_threads / 4)
        self.train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers, collate_fn=collate_fn)
        self.dev_loader = data.DataLoader(dataset=dev_set, batch_size=batch_size_test, shuffle=False,
                                          num_workers=num_workers, collate_fn=collate_fn)
        self.test_loader = data.DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=False,
                                           num_workers=num_workers, collate_fn=collate_fn)


class GSCDDataset(data.Dataset):
    def __init__(self, proj: Project,
                 feat_path: str):
        """
        param name: 'train', 'dev' or 'test'
        """
        # Load data
        with h5py.File(feat_path, 'r') as hf:
            # Load H5PY Data
            dict_data = load_h5py_data(hf)
            self.features = dict_data['features'].astype(np.float32)
            self.feature_lengths = dict_data['feature_lengths'].astype(int)
            self.targets = np.squeeze(dict_data['targets'].astype(int))
            self.target_lengths = dict_data['target_lengths'].astype(int)
            self.flags = dict_data['flag'].astype(int)
            self.num_features = int(dict_data['n_features'].astype(int))
            self.num_classes = int(dict_data['n_classes'].astype(int))
            self.stat = {}
            self.num_samples = len(self.feature_lengths)
            self.feature_slices = idx_to_slice(self.feature_lengths)
            self.target_slices = idx_to_slice(self.target_lengths)

            # Numpy Arrays to PyTorch Tensors
            self.features = torch.tensor(self.features).float()
            self.feature_lengths = torch.tensor(self.feature_lengths).long()
            self.targets = torch.tensor(self.targets).long()
            self.target_lengths = torch.tensor(self.target_lengths).long()
            self.flags = torch.tensor(self.flags).long()

        # Update arguments
        proj.additem('input_size', self.num_features)
        proj.additem('num_classes', self.num_classes)

    def __len__(self):
        'Total number of samples'
        return self.num_samples  # The first dimention of the data tensor

    def __getitem__(self, idx):
        'Get one sample from the dataset using an index'
        feature = self.features[slc(idx, self.feature_slices), :]
        feature_length = self.feature_lengths[idx]
        target = self.targets[slc(idx, self.target_slices)]
        target_length = self.target_lengths[idx]
        flag = self.flags[idx]

        return feature, feature_length, target, target_length, flag


def slc(idx, slices):
    return slice(slices[idx][0], slices[idx][1])


def log_lut(x, qi_in, qf_in, qi_out, qf_out, en, approx):
    if en:
        if approx:
            lut_in = quantize_array(x, qi_in, qf_in, 1)
            lut_out = np.log10(lut_in + 1)
            lut_out = quantize_array(lut_out, qi_out, qf_out, 1)
        else:
            lut_out = np.log10(1 + x)
        return lut_out
    else:
        return x


def quantize_feature(features, pre_max, fqi, fqf, en):
    if en:
        max_dynamic = 2 ** (fqi - 1)
        # Map Features to [0, 1]
        features /= pre_max
        # Scale Features to max dynamic range
        features = np.floor(features * max_dynamic)  # correct
        features = quantize_array(features, fqi, fqf, 1)
    return features


def process_train(proj: Project, train_set):
    stat = {'pre_mean_feat': torch.mean(train_set.features.view(-1, train_set.num_features), dim=0),
            'pre_std_feat': torch.std(train_set.features.view(-1, train_set.num_features), dim=0),
            'pre_max_feat': torch.amax(train_set.features.view(-1, train_set.num_features), dim=0),
            'pre_min_feat': torch.amin(train_set.features.view(-1, train_set.num_features), dim=0),
            'pre_shape': train_set.features.shape,
            'pre_mean': torch.mean(train_set.features),
            'pre_std': torch.std(train_set.features),
            'pre_max': torch.amax(train_set.features),
            'pre_min': torch.amin(train_set.features),
            'pre_num_sample': train_set.features.size(0)}

    if proj.qf:
        train_set.features = quantize_feature(train_set.features, stat['pre_max'], proj.fqi, proj.fqf, proj.qf)
    if proj.logf == 'lut':
        train_set.features = log_lut(train_set.features, qi_in=proj.fqi, qf_in=proj.fqf, qi_out=3, qf_out=8,
                                     en=proj.log_feat, approx=proj.approx_log)

    stat['post_mean_feat'] = torch.mean(train_set.features.view(-1, train_set.num_features), dim=0)
    stat['post_std_feat'] = torch.std(train_set.features.view(-1, train_set.num_features), dim=0)
    stat['post_max_feat'] = torch.amax(train_set.features.view(-1, train_set.num_features), dim=0)
    stat['post_min_feat'] = torch.amin(train_set.features.view(-1, train_set.num_features), dim=0)
    stat['post_shape'] = train_set.features.shape
    stat['post_mean'] = torch.mean(train_set.features)
    stat['post_std'] = torch.std(train_set.features)
    stat['post_max'] = torch.amax(train_set.features)
    stat['post_min'] = torch.amin(train_set.features)
    stat['post_num_sample'] = train_set.features.size(0)

    if proj.norm_feat:
        train_set.features -= stat['post_mean']
        train_set.features /= stat['post_std']

    return train_set, stat


def process_test(proj: Project, test_set, train_set_stat):
    stat = {'pre_mean_feat': torch.mean(test_set.features.view(-1, test_set.num_features), dim=0),
            'pre_std_feat': torch.std(test_set.features.view(-1, test_set.num_features), dim=0),
            'pre_max_feat': torch.amax(test_set.features.view(-1, test_set.num_features), dim=0),
            'pre_min_feat': torch.amin(test_set.features.view(-1, test_set.num_features), dim=0),
            'pre_shape': test_set.features.shape,
            'pre_mean': torch.mean(test_set.features),
            'pre_std': torch.std(test_set.features),
            'pre_max': torch.amax(test_set.features),
            'pre_min': torch.amin(test_set.features),
            'pre_num_sample': test_set.features.size(0)}

    if proj.qf:
        test_set.features = quantize_feature(test_set.features, train_set_stat['pre_max'], proj.fqi, proj.fqf, proj.qf)
    if proj.logf == 'lut':
        test_set.features = log_lut(test_set.features, qi_in=proj.fqi, qf_in=proj.fqf, qi_out=3, qf_out=8,
                                    en=proj.log_feat, approx=proj.approx_log)

    stat['post_mean_feat'] = torch.mean(test_set.features.view(-1, test_set.num_features), dim=0)
    stat['post_std_feat'] = torch.std(test_set.features.view(-1, test_set.num_features), dim=0)
    stat['post_max_feat'] = torch.amax(test_set.features.view(-1, test_set.num_features), dim=0)
    stat['post_min_feat'] = torch.amin(test_set.features.view(-1, test_set.num_features), dim=0)
    stat['post_shape'] = test_set.features.shape
    stat['post_mean'] = torch.mean(test_set.features)
    stat['post_std'] = torch.std(test_set.features)
    stat['post_max'] = torch.amax(test_set.features)
    stat['post_min'] = torch.amin(test_set.features)
    stat['post_num_sample'] = test_set.features.size(0)

    if proj.norm_feat:
        test_set.features -= train_set_stat['post_mean']
        test_set.features /= train_set_stat['post_std']

    return test_set, stat


def idx_to_slice(lengths):
    """
    Get the index range of samples
    :param lengths: 1-D tensor containing lengths in time of each sample
    :return: A list of tuples containing the start & end indices of each sample
    """
    idx = []
    lengths_cum = np.cumsum(lengths)
    for i, len in enumerate(lengths):
        start_idx = lengths_cum[i] - lengths[i]
        end_idx = lengths_cum[i]
        idx.append((start_idx, end_idx))
    return idx



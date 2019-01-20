# __author__ = "Chang Gao"
# __copyright__ = "Copyright 2018 to the author"
# __license__ = "Private"
# __version__ = "0.1.0"
# __maintainer__ = "Chang Gao"
# __email__ = "chang.gao@uzh.ch"    `
# __status__ = "Prototype"

import os
import ast
import math
import pandas as pd
from tqdm import tqdm
import h5py
from modules.feat import get_wav_file, compute_fbank, write_hdf5
import numpy as np
import errno


def main(root, nfilt=40, frame_size=0.1, frame_stride=0.1, window_size=5, MFCC=False):
    print("####################################################################")
    print("# VAD Step 1: Feature Extraction                                   #")
    print("####################################################################")

    try:
        os.makedirs(os.path.join(root, 'feat'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Load dataframe
    df = pd.read_csv(os.path.join(root, 'data', 'description_qut.csv'))

    # Initialization
    np.random.seed(0)

    X_train = np.zeros(1, dtype=np.int32)
    y_train = np.zeros(1, dtype=np.int32)
    X_test = np.zeros(1, dtype=np.int32)
    y_test = np.zeros(1, dtype=np.int32)

    # Loop over dataframe
    n_train = 0
    n_test = 0
    n_row = 0

    selected_rows = []

    for row in (df.itertuples()):
        n_row = n_row + 1
        if (row.snr == 'n+10' or row.snr == 'n+15'):# and ('CAFE' in row.category):
            selected_rows.append((row.path, row.timestep, row.label, row.group, row.snr))

    for row in tqdm(selected_rows):
        # Get features
        sample, sample_rate = get_wav_file(row[0])
        feature = compute_fbank(signal=sample, sample_rate=sample_rate, frame_size=frame_size, frame_stride=frame_stride, nfilt=nfilt, MFCC=MFCC)

        # Create labels
        len_feat = feature.shape[0]
        label = np.zeros(len_feat, dtype=np.int32)
        timestep = np.array(ast.literal_eval(row[1]), dtype=np.float32)
        prelabel = np.array(ast.literal_eval(row[2]), dtype=np.int32)
        prev_frame_idx = 0
        for idx, step in enumerate(timestep):
            frame_idx = int(math.floor(step - frame_size) / frame_stride + 1)
            label[prev_frame_idx:frame_idx] = prelabel[idx]
            prev_frame_idx = frame_idx

        # Combine every N frames (N must be an odd number)
        N = window_size
        edge_offset = int((N + 1)/2 - 1)
        label = label[edge_offset:feature.shape[0] - edge_offset]  # Reshape labels
        new_feature = []
        for i in range(edge_offset, feature.shape[0]-edge_offset):
            new_feature.append(feature[int(i - (N - 1)/2):int(i + (N - 1)/2) + 1, :].reshape(-1))
        feature = np.vstack(new_feature)

        if row[3] == 'a':
            if n_train == 0:
                X_train = feature
                y_train = label
                n_train = n_train + 1
            else:
                X_train = np.vstack((X_train, feature))
                y_train = np.hstack((y_train, label))
        elif row[3] == 'b':
            if n_test == 0:
                X_test = feature
                y_test = label
                n_test = n_test + 1
            else:
                X_test = np.vstack((X_test, feature))
                y_test = np.hstack((y_test, label))

    with h5py.File(os.path.join(root, 'feat', 'train_vad.h5'), 'w') as f:
        f.create_dataset('features', data=X_train)
        f.create_dataset('labels', data=y_train)

    with h5py.File(os.path.join(root, 'feat', 'test_vad.h5'), 'w') as f:
        f.create_dataset('features', data=X_test)
        f.create_dataset('labels', data=y_test)

    print("Feature stored in: ", os.path.join(root, 'feat'))
    print("Feature Extraction Completed...                                     ")
    print(" ")

# __author__ = "Chang Gao"
# __copyright__ = "Copyright 2018 to the author"
# __license__ = "Private"
# __version__ = "0.1.0"
# __maintainer__ = "Chang Gao"
# __email__ = "chang.gao@uzh.ch"    `
# __status__ = "Prototype"

import os
import ast
import pandas as pd
from tqdm import tqdm
from modules.feat import get_wav_file, compute_fbank, write_hdf5
import numpy as np
import errno


def main(root, nfilt=40, frame_size=0.025, frame_stride=0.01, MFCC=False, phn=48):
    print("####################################################################")
    print("# ASR Step 1: Feature Extraction                                   #")
    print("####################################################################")

    try:
        os.makedirs(os.path.join(root, 'feat'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Load dataframe
    df = pd.read_csv(os.path.join(root, 'data', 'description.csv'))

    # Initialization
    np.random.seed(2)

    X_train = []
    X_dev = []
    X_test = []

    y_train = []
    y_dev = []
    y_test = []

    # Loop over dataframe
    for row in tqdm(df.itertuples()):

        # Get features
        sample, sample_rate = get_wav_file(row.path)

        feature = compute_fbank(signal=sample, sample_rate=sample_rate, frame_size=frame_size, frame_stride=frame_stride, nfilt=nfilt, MFCC=MFCC)

        if (row.group == 'train'):
            X_train.append(feature.T)
            if phn == 48:
                y_train.extend([ast.literal_eval(row.label48)])
            elif phn == 61:
                y_train.extend([ast.literal_eval(row.label61)])
        elif (row.group == 'dev'):
            X_dev.append(feature.T)
            if phn == 48:
                y_dev.extend([ast.literal_eval(row.label48)])
            elif phn == 61:
                y_dev.extend([ast.literal_eval(row.label61)])
        elif (row.group == 'test'):
            X_test.append(feature.T)
            y_test.extend([ast.literal_eval(row.label39)])

    # get feature_lens
    write_hdf5(os.path.join(root, 'feat', 'train.h5'), X_train, y_train)
    write_hdf5(os.path.join(root, 'feat', 'dev.h5'), X_dev, y_dev)
    write_hdf5(os.path.join(root, 'feat', 'test.h5'), X_test, y_test)

    print("Feature stored in: ", os.path.join(root, 'feat'))
    print("Feature Extraction Completed...                                     ")
    print(" ")

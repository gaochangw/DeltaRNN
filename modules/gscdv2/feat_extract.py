import math
import os
import numpy as np
import pandas as pd
from utils.feature.logfilterbank import logfbank
from utils.feature.iir import iir
from tqdm import tqdm
from utils.util_feat import write_dataset
import soundfile as snd


class FeatExtractor:
    def __init__(self, proj):
        if proj.augment_noise:
            self.description_file = 'description_snr' + '.csv'
        else:
            self.description_file = 'description.csv'

        # Load Configurations
        for k, v in proj.config.items():
            setattr(self, k, v)

        # Load Description File
        self.df_description = pd.read_csv(
            os.path.join('data', proj.dataset, self.description_file))

    def extract(self, proj):
        X_train = []
        X_val = []
        X_test = []

        f_train = []
        f_val = []
        f_test = []

        y_train = []
        y_val = []
        y_test = []

        # Get File ID
        _, trainfile, devfile = proj.log.gen_trainset_name(proj)
        testfile = proj.log.gen_testset_name(proj)

        # Get Training Set WAV Statistics
        audio_signal_classes = [[] for i in range(0, 2)]
        audio_min = np.zeros(2)
        audio_max = np.zeros(2)
        audio_mean = np.zeros(2)
        audio_std = np.zeros(2)
        audio_drange = np.zeros(2)
        for row in tqdm(self.df_description.itertuples(), total=self.df_description.shape[0]):
            if row.group == 'train':
                filepath = row.path
                filepath = filepath.replace("\\", "/")
                filepath = filepath.replace("C:/", "/")
                audio_signal, sample_rate = snd.read(filepath, dtype='int16')
                if int(row.label) == 0:
                    audio_signal_classes[0].append(audio_signal)
                else:
                    audio_signal_classes[1].append(audio_signal)

        for i in range(0, 2):
            audio_signal_classes[i] = np.concatenate(audio_signal_classes[i])
            audio_signal_classes[i] = audio_signal_classes[i].astype(np.float64)
            audio_min[i] = np.amin(audio_signal_classes[i])
            audio_max[i] = np.amax(audio_signal_classes[i])
            audio_mean[i] = np.mean(audio_signal_classes[i])
            audio_std[i] = np.std(audio_signal_classes[i])

        keyword_mean = audio_mean[1]
        keyword_std = audio_std[1]
        keyword_drange = 5*keyword_std
        # print(audio_signal_classes)
        # audio_signal /= drange
        # audio_signal *= 2 ** (self.mic_res - 1)

        # audio_min = np.amin(audio_signal)
        # audio_max = np.amax(audio_signal)
        # audio_ = np.amax(audio_signal)
        # audio_max = np.amax(audio_signal)
        # drange = max(np.abs(audio_min), np.abs(audio_max))
        # audio_signal = audio_signal.astype(np.float64)
        # audio_signal /= drange
        # audio_signal *= 2 ** (mic_res - 1)

        # Loop over dataframe
        for row in tqdm(self.df_description.itertuples(), total=self.df_description.shape[0]):
            filepath = row.path
            filepath = filepath.replace("\\", "/")
            filepath = filepath.replace("C:/", "/")

            # Digital Audio Front End (Log Filter Bank)
            if self.feat_type == 'logfbank':
                # Feature
                features, frames, sample_rate = logfbank(path=filepath,
                                                         n_filt=self.n_filt,
                                                         f_spacing=self.f_spacing,
                                                         f_centers=self.f_centers,
                                                         freq_low=self.freq_low,
                                                         freq_high=self.freq_high,
                                                         frame_size=self.frame_size,
                                                         frame_stride=self.frame_stride,
                                                         oversample_factor=self.oversample_factor,
                                                         nfft=self.nfft,
                                                         mfcc=self.mfcc,
                                                         gain_ctrl=self.gain_ctrl,
                                                         gradient=self.gradient,
                                                         plot=0)

                actual_n_frames = features.shape[0]
                label = np.zeros((actual_n_frames, 1))
                label[self.label_head:self.label_tail, :] = int(row.label)
                flag = int(row.label)

                # Use VAD to label the samples
                # if self.use_vad:
                #     import webrtcvad
                #     vad = webrtcvad.Vad()
                #     vad.set_mode(3)
                #
                #     label = np.zeros((actual_n_frames,))
                #     for i in range(actual_n_frames):
                #         label[i] = int(vad.is_speech(frames[i, :], int(sample_rate)))
                #     pos_label_idx = np.argwhere(label == 1)
                #     len_voice = pos_label_idx.shape[0]
                #     len_label = int(math.floor(len_voice * 0.9))
                #     if len_voice != 0:
                #         dec_window_start_idx = int(pos_label_idx[-1] - self.delta_t)
                #         # dec_window_start_idx = int(pos_label_idx[0])
                #         dec_window_end_idx = int(pos_label_idx[-1])
                #         # dec_window_end_idx = int(pos_label_idx[-1] + delta_t)
                #         label = np.zeros((actual_n_frames, 1))
                #         # print(int(row.label))
                #         label[dec_window_start_idx:dec_window_end_idx, :] = int(row.label) + 1
                #         flag = int(row.label) + 1
                #     else:
                #         continue
                #         # label = np.zeros((dim_T, 1))
                #         # flag = 0

            elif self.feat_type == 'iir':
                features, sample_rate = iir(path=filepath,
                                            sim_mic=self.sim_mic,
                                            mic_res=self.mic_res,
                                            drange=keyword_drange,
                                            mean=keyword_mean,
                                            n_filt=self.n_filt,
                                            f_spacing=self.f_spacing,
                                            f_centers=self.f_centers,
                                            freq_low=self.freq_low,
                                            freq_high=self.freq_high,
                                            frame_size=self.frame_size,
                                            frame_stride=self.frame_stride,
                                            oversample_factor=self.oversample_factor,
                                            q_factors=self.q_factors,
                                            ch_orders=self.ch_orders,
                                            plot=0)

                dim_T = features.shape[0]
                label = np.zeros((dim_T, 1))
                # if np.max(features) > 0:
                label[self.label_head:self.label_tail, :] = int(row.label)
                flag = int(row.label)
                # else:
                #     flag = 0
            else:
                raise RuntimeError('Please select a valid feature type.')

            if row.group == 'train':
                X_train.append(features)
                y_train.append(label)
                f_train.append(flag)
            elif row.group == 'val':
                X_val.append(features)
                y_val.append(label)
                f_val.append(flag)
            elif row.group == 'test':
                X_test.append(features)
                y_test.append(label)
                f_test.append(flag)

        # Process Datasets
        dataset_train = self.get_dataset(X_train, y_train, f_train)
        dataset_val = self.get_dataset(X_val, y_val, f_val)
        dataset_test = self.get_dataset(X_test, y_test, f_test)

        # Write Dataset
        write_dataset(os.path.join('feat', proj.dataset, trainfile), dataset_train)
        write_dataset(os.path.join('feat', proj.dataset, devfile), dataset_val)
        write_dataset(os.path.join('feat', proj.dataset, testfile), dataset_test)

        print("Feature stored in: ", 'feat')
        print("Feature Extraction Completed...                                     ")
        print(" ")

    # def get_dataset(self, x, y, f):
    #     features = np.stack(x, axis=0).astype(np.float32)
    #     targets = np.stack(y, axis=0).astype(np.int32)
    #     flag = np.stack(f, axis=0).astype(np.int32)
    #     n_features = features.shape[-1]
    #     n_classes = np.max(flag).astype(np.int32) + 1
    #     dict_dataset = {'features': features, 'targets': targets, 'flag': flag, 'n_features': n_features,
    #                     'n_classes': n_classes}
    #     return dict_dataset

    def get_dataset(self, x, y, f):
        features = np.concatenate(x, axis=0).astype(np.float32)
        feature_lengths = np.asarray([sample.shape[0] for sample in x]).astype(np.int64)
        targets = np.concatenate(y, axis=0).astype(np.int64)
        target_lengths = np.asarray([len(target) for target in y]).astype(np.int64)
        flag = np.stack(f, axis=0).astype(np.int64)
        n_features = features.shape[-1]
        n_classes = np.max(flag).astype(np.int64) + 1
        dict_dataset = {'features': features, 'feature_lengths': feature_lengths, 'targets': targets,
                        'target_lengths': target_lengths, 'flag': flag, 'n_features': n_features,
                        'n_classes': n_classes}
        return dict_dataset


def align(x):
    aligned = []
    len = x.shape[1]
    for i in x:
        all_mean = []
        for t in range(0, len - 61):
            sec = i[t:t + 61, :]
            all_mean.append(np.mean(sec))
        all_mean = np.asarray(all_mean)
        start_index = np.argmax(all_mean)
        aligned.append(i[start_index:start_index + 61])
    aligned = np.stack(aligned, axis=0)
    return aligned

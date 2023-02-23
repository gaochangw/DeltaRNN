__author__ = "Kwantae Kim, Chang Gao"
__copyright__ = "Copyright 2020"
# __credits__ = [""]
__license__ = "Private"
__version__ = "1.0.6"
__maintainer__ = "Kwantae Kim, Chang Gao"
__email__ = "chang.gao@uzh.ch"
__status__ = "Prototype"

import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as snd


def extract_feat(path,
                 config,
                 hz_points,
                 filt_b,
                 filt_a,
                 plot=0):
    # Hyperparameters
    mode = config['feature']['aafe']['mode']
    n_filt = config['feature']['n_filt']
    frame_size = config['feature']['frame_size']
    frame_stride = config['feature']['frame_stride']
    oversample_factor = config['feature']['oversample_factor']

    # Read WAV file
    audio_signal, sample_rate = snd.read(path, dtype='int16')

    # Oversampling
    if oversample_factor > 1:
        audio_signal = signal.resample(audio_signal, audio_signal.shape[0] * oversample_factor)
        sample_rate *= oversample_factor

    # Apply the BPF
    signal_ch = []
    for i in range(n_filt):
        # Apply a digital filter forward and backward to a signal.
        # This function applies a linear digital filter twice, once forward and once backwards.
        # The combined filter has zero phase and a filter order twice that of the original.
        # signal_filtered = signal.filtfilt(b=filt_b[i],  # The numerator coefficient vector of the filter.
        #                                         a=filt_a[i],  # The denominator coefficient vector of the filter.
        #                                         x=signal)  # The array of data to be filtered.
        signal_filtered = signal.lfilter(b=filt_b[i],  # The numerator coefficient vector of the filter.
                                         a=filt_a[i],  # The denominator coefficient vector of the filter.
                                         x=audio_signal)  # The array of data to be filtered.
        signal_ch.append(signal_filtered)
    signal_ch = np.stack(signal_ch, axis=0)

    # Save Channel 0 to file
    np.savetxt('yes.txt', signal_ch[0], delimiter=' ')

    # Framing
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = signal_ch.shape[1]
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.zeros((n_filt, pad_signal_length))
    pad_signal[:, :signal_length] = signal_ch  # Pad Signal to make sure that all frames have equal number of samples

    # without truncating any samples from the original signal
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    signal_frames_ch = pad_signal[:, indices.astype(np.int32, copy=False)]  # Shape: (N_FILT, N_FRAME, N_SAMPLES)

    # Derive the averaged amplitude per fram
    if mode == 'fwrmag':  # Full wave rectifier
        rectified = np.absolute(signal_frames_ch)
        features = np.mean(rectified, axis=-1)  # Shape: (N_FILT, N_FRAME)
        features = np.where(features == 0, np.finfo(float).eps, features)  # Numerical Stability

        # # mean_feat = np.mean(features)
        # # std_feat = np.std(features)
        #
        # # Remove outliers of Analog AFE features by clipping at upper 2-Sigma
        # # features = np.clip(features, 0, mean_feat + 5 * std_feat)
        #
        # # Get Range of Quantized Features
        # # processed_features_max = np.amax(features)
        # processed_features_max = 16000
        #
        # # Map Features to [0, 1]
        # features /= processed_features_max
        #
        # # Quantize Features to Fixed Point Numbers
        # features = np.around(features * (2 ** 12 - 1)) + 1
        # features = np.where(features == 0, np.finfo(float).eps, features)  # Numerical Stability
        # features = np.log(features + 1)
    elif mode == 'fwrpwr':  # Power
        rectified = np.square(signal_frames_ch)
        features = np.mean(rectified, axis=-1)  # Shape: (N_FILT, N_FRAME)
        features = np.where(features == 0, np.finfo(float).eps, features)  # Numerical Stability
    elif mode == 'hwr':  # Positive Half wave rectifier
        rectified = np.maximum(0, signal_frames_ch)
        features = np.mean(rectified, axis=-1)  # Shape: (N_FILT, N_FRAME)
        features = np.where(features == 0, np.finfo(float).eps, features)  # Numerical Stability
    elif mode == 'hwrcat':  # Postive + Negative half wave rectifier
        rectified_pos = np.maximum(0, signal_frames_ch)
        rectified_neg = np.minimum(0, signal_frames_ch)
        rectified = np.concatenate((rectified_pos, rectified_neg), axis=0)
        features = np.mean(rectified, axis=-1)  # Shape: (N_FILT, N_FRAME)
        features = np.where(features == 0, np.finfo(float).eps, features)  # Numerical Stability
    else:
        raise ValueError('Mode not supported.')

    # Take the mean of all samples in each frame
    # features = np.mean(rectified, axis=-1)  # Shape: (N_FILT, N_FRAME)
    # features = np.where(features == 0, np.finfo(float).eps, features)  # Numerical Stability
    features = features.T

    # Plot the filters
    if plot:
        # Plot Frame
        # wave = frames_sample_ch[0, 50, :]
        # full_wave = np.absolute(wave)
        # fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        # axes[0].plot(wave, '-o', linewidth=2, markersize=4)
        # axes[0].set_xlabel('Time', fontsize=18)
        # axes[0].set_ylabel('Amplitude', fontsize=18)
        # axes[0].tick_params(labelsize=16)
        # axes[0].grid(which='both', axis='both')
        # axes[1].plot(full_wave, '-o', linewidth=2, markersize=4)
        # axes[1].set_xlabel('Time', fontsize=18)
        # axes[1].set_ylabel('Amplitude', fontsize=18)
        # axes[1].tick_params(labelsize=16)
        # axes[1].grid(which='both', axis='both')

        # Original audio wave
        time = np.linspace(0, audio_signal.shape[0] / sample_rate, audio_signal.shape[0])
        fig, axes = plt.subplots(1, 1, figsize=(24, 8), sharex='row')
        fig.suptitle('Original Wave | Sample: Bed', fontsize=18, fontweight='bold')
        axes.plot(time, audio_signal, zorder=1, label='Original Wave')
        axes.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
        axes.set_xlim(0.5, 0.8)
        plt.subplots_adjust(top=0.92, bottom=0.1, left=0.09, right=0.95, wspace=0.8, hspace=0.8)

        # Plot filtered audio wave vs. original audio wave
        time = np.linspace(0, audio_signal.shape[0] / sample_rate, audio_signal.shape[0])
        fig, axes = plt.subplots(8, 1, figsize=(24, 14), sharex='row')
        fig.suptitle('Sample: Bed | Feature Type: BPF | Filter Spacing: Mel | Freq Range: 100 Hz - 8000 Hz',
                     fontsize=18, fontweight='bold')
        for i in range(8):
            axes[i].plot(time, audio_signal, zorder=1, label='Original Wave')
            axes[i].plot(time, signal_ch[i * 2, :], zorder=2, label='Filtered Wave')
            axes[i].set_xlim(0.5, 0.8)
            plt_title = 'Filter: #' + str(i * 2) + ' | Center Freq: ' + f"{hz_points[i * 2]:.2f}"
            axes[i].set_title(plt_title, fontsize=16, fontweight='bold')

        axes[7].set_xlabel('Time (s)', fontsize=16, fontweight='bold')
        axes[0].legend(loc=2, fontsize=14)
        plt.subplots_adjust(top=0.92, bottom=0.05, left=0.09, right=0.95, wspace=0.8, hspace=0.8)
        # Plot the feature vector
        # plt.figure()
        # for i in range(6):
        #     plt.subplot(6, 1, i + 1)
        #     plt.plot(features[i * 3])
        #     plt_xlabel = 'Frame Window [' + str(frame_size) + 'ms]'
        #     plt.xlabel(plt_xlabel)
        #     plt_title = 'Feature Vector Ch' + str(i * 3)
        #     plt.title(plt_title)
        #     plt.ylim = (0, 256)

        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        im1 = axes.imshow(features.T, aspect='auto')
        plt.colorbar(im1)
        axes.tick_params(labelsize=16)
        axes.set_xlabel('Time (s)', fontsize=18)
        axes.set_ylabel('Channels', fontsize=18)
        plt.show()

    # print('The dtype of output feature vector (f_vector) is', f_vector.dtype)
    return features

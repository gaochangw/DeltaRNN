import scipy
from scipy.fftpack import dct
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as snd


def framing(path, frame_size=0.025, frame_stride=0.01, oversample_factor=0):
    # Initialization
    np.random.seed(2)
    from scipy.io import wavfile
    # sample is a 1-dim vector
    # sample_rate, sample = wavfile.read(path)

    # Ignore
    # def gen_new_path(path):
    #     filename = path.split('.wav')[0]
    #     new_filename = filename + '_norm.wav'
    #     return new_filename
    # path = gen_new_path(path)

    audio_signal, sample_rate = snd.read(path, dtype='int32')

    # Oversample
    if oversample_factor > 1:
        audio_signal = signal.resample(audio_signal, audio_signal.shape[0] * oversample_factor)
        sample_rate *= oversample_factor

    sample_rate = float(sample_rate)

    # Framing
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(audio_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(audio_signal,
                           z)  # Pad Signal to make sure that all frames have equal number of samples without
    # truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    frames = pad_signal[indices.astype(np.int32, copy=False)]

    return frames, num_frames, frame_length, sample_rate


def extract_feat(path,
                 config,
                 fbank,
                 NFFT=512,
                 MFCC=0,
                 plot=0):
    # Hyperparameters
    frame_size = config['feature']['frame_size']
    frame_stride = config['feature']['frame_stride']
    oversample_factor = config['feature']['oversample_factor']
    NFFT = config['feature']['dafe']['nfft']
    gain_ctrl = config['feature']['gain_ctrl']
    gradient = config['feature']['gradient']

    # Get Frames
    frames, num_frames, frame_length, sample_rate = framing(path, frame_size, frame_stride, oversample_factor)

    # Real-Time Normalization (Volumn Control)
    processed_frames = frames
    if gain_ctrl:
        batch_idx = 0
        batch_size = 8
        norm_frames = np.asarray([]).reshape(0, frames.shape[1])
        num_batch = np.ceil(num_frames / float(batch_size)).astype(int)

        for i in range(0, num_batch):
            if i == (num_batch - 1):
                batch_curr = frames[batch_size * i:, :]
            else:
                batch_curr = frames[batch_size * i:batch_size * (i + 1), :]
            stack_curr = batch_curr.reshape(-1)
            mean = np.mean(stack_curr)
            std = np.std(stack_curr, ddof=1)
            batch_curr -= mean
            if std != 0:
                batch_curr /= std
            norm_frames = np.vstack((norm_frames, batch_curr))

        processed_frames = norm_frames

    # Windowing
    processed_frames *= np.hamming(frame_length)
    dim_T = processed_frames.shape[0]

    # Apply Filters to Frames
    # Fourier-Transform and Power Spectrum
    mag_frames = np.absolute(np.fft.rfft(processed_frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))  # Power Spectrum
    mag_frames /= float(NFFT)
    feat_frames = mag_frames

    # energy = np.sum(feat_frames, axis=-1, keepdims=True)
    features = np.dot(feat_frames, fbank.T)
    features = np.where(features == 0, np.finfo(float).eps, features)  # Numerical Stability
    # features = np.log(features + 1)  # dB
    # features = np.hstack((features, energy))
    if gradient:
        features_grad_first_order = np.gradient(features, edge_order=2, axis=0)
        features_grad_second_order = np.gradient(features_grad_first_order, edge_order=2, axis=0)
        features = np.hstack((features, features_grad_first_order, features_grad_second_order))

    if (MFCC == 1):
        num_ceps = 39
        cep_lifter = 22
        mfcc = dct(features, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13
        (nframes, ncoeff) = mfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift  # *
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
        features = mfcc

    if plot:
        # freq_step = sample_rate / 2 / (NFFT / 2)
        # freq = np.arange(0, int(NFFT / 2) + 1) * freq_step
        # fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        # axes[0].plot(freq, mag_frames[0, :], '-o', linewidth=3, markersize=8)
        # axes[0].set_xlabel('Frequency', fontsize=18)
        # axes[0].set_ylabel('Magnitude', fontsize=18)
        # axes[0].tick_params(labelsize=16)
        # axes[1].tick_params(labelsize=16)
        # axes[1].plot(freq, pow_frames[0, :], '-o', linewidth=2, markersize=8)
        # axes[1].set_xlabel('Frequency', fontsize=18)
        # axes[1].set_ylabel('Power', fontsize=18)
        # axes[0].grid(which='both', axis='both')
        # axes[1].grid(which='both', axis='both')

        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        im1 = axes.imshow(features.T, aspect='auto')
        plt.colorbar(im1)
        axes.tick_params(labelsize=16)
        axes.set_xlabel('Time (s)', fontsize=18)
        axes.set_ylabel('Channels', fontsize=18)
        # axes[0].set_xlim([0, 260])
        # axes[0].set_ylim([0, 39])
        # axes[0].set_yticks(range(0, 50, 10))
        # axes[0].set_xticks(np.arange(0, 270, 20))
        plt.show()

    return features, frames, sample_rate


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(window_size * sample_rate / 1e3)
    noverlap = int(step_size * sample_rate / 1e3)
    _, _, spec = scipy.signal.spectrogram(audio,
                                          fs=sample_rate,
                                          window='hann',
                                          nperseg=nperseg,
                                          noverlap=noverlap,
                                          detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)

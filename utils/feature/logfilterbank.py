import scipy
from scipy.fftpack import dct
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from utils.util_feat import mel_space_dafe, log_space_dafe


def logfbank(path,
             n_filt,
             f_spacing,
             freq_low,
             freq_high,
             f_centers,
             frame_size,
             frame_stride,
             oversample_factor,
             nfft=512,
             mfcc=0,
             gain_ctrl=0,
             gradient=0,
             plot=0):
    #############################
    # Framing
    #############################

    # Get Frames
    import soundfile as snd
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

    #############################
    # Build Filters
    #############################

    # Set Upper Frequency
    if freq_high is None:
        freq_high = sample_rate / 2 - 500

    # Filter Banks
    if f_spacing == 'mel':
        hz_points = mel_space_dafe(freq_low, freq_high, n_filt)
    elif f_spacing == 'log':
        hz_points = log_space_dafe(freq_low, freq_high, n_filt)
    else:
        hz_points = np.asarray(f_centers)

    # Create Filters
    # Limit Hz Point between [0, sample_rate/2]
    # hz_points = np.clip(hz_points, 0, sample_rate / 2)
    bin = np.floor((nfft + 1) * hz_points / sample_rate)


    # Plot
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        freq_step = sample_rate / 2 / (nfft / 2)
        freq = np.arange(0, int(nfft / 2) + 1) * freq_step

    fbank = np.zeros((n_filt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, n_filt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        if plot:
            # axes.semilogx((filt_w[i] / (2 * np.pi)) * sample_rate, 20 * np.log10(abs(filt_h[i])))
            axes[0].semilogx(freq, 20 * np.log10(abs(fbank[m - 1])))
            axes[0].set_xlim([1e1, 1e5])
            axes[0].set_xlabel('log(frequency)', fontsize=18)
            axes[0].set_ylabel('dB', fontsize=18)
            axes[0].tick_params(labelsize=16)
            axes[1].plot(freq, fbank[m - 1])
            axes[1].set_xlabel('Frequency', fontsize=18)
            axes[1].set_ylabel('Amplitude Gain', fontsize=18)
            axes[1].tick_params(labelsize=16)
            axes[1].set_xlim(0, 16000)
    if plot:
        axes[0].grid(which='both', axis='both')
        axes[1].grid(which='both', axis='both')
        plt.show()


    #############################
    # Gain Control (Normalization)
    #############################
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
    mag_frames = np.absolute(np.fft.rfft(processed_frames, nfft))  # Magnitude of the FFT
    pow_frames = ((1.0 / nfft) * (mag_frames ** 2))  # Power Spectrum
    mag_frames /= float(nfft)
    feat_frames = mag_frames

    # energy = np.sum(feat_frames, axis=-1, keepdims=True)
    features = np.dot(feat_frames, fbank.T)
    features = np.where(features == 0, np.finfo(float).eps, features)  # Numerical Stability
    features = 20*np.log10(features)  # dB
    # features = np.hstack((features, energy))
    if gradient:
        features_grad_first_order = np.gradient(features, edge_order=2, axis=0)
        features_grad_second_order = np.gradient(features_grad_first_order, edge_order=2, axis=0)
        features = np.hstack((features, features_grad_first_order, features_grad_second_order))

    if mfcc:
        num_ceps = 39
        cep_lifter = 22
        features = dct(features, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13
        (nframes, ncoeff) = features.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        features *= lift  # *
        features -= (np.mean(features, axis=0) + 1e-8)

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

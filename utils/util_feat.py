import platform
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import h5py


# SNR mixer
def add_noise(signal, noise, snr):
    """
    :param signal: Signal to be mixed with noise
    :param noise: Noise
    :param snr: Signal-to-Noise Ratio
    :return:
    >>> xrange = np.linspace(0,10,101)
    >>> sig = np.sin(xrange)
    >>> noi = 0.1*np.sin(5*xrange)
    >>> out = add_noise(sig,noi,10)
    >>> np.sum(out)
    13.826366197746081
    >>> xrange = np.linspace(0,10,101)
    >>> sig = np.sin(xrange)
    >>> noi = 0.1*np.sin(35.1*xrange)
    >>> out = add_noise(sig,noi,4.74)
    >>> np.sum(out)
    11.668733164395023
    """
    signal = np.asarray(signal)
    noise = np.asarray(noise)
    snr = float(snr)

    # Get amplification factor for signal. Using power snr method
    log_amp_factor = snr / 10.0 + np.log10(np.mean(np.square(noise))) - np.log10(np.mean(np.square(signal)))
    amp_factor = np.sqrt(10.0 ** log_amp_factor)

    # Get output signal and normalize
    out = signal * amp_factor + noise
    out = out / np.max(np.abs(out))
    out = out * 0.99  # avoid clipping

    return out


def write_dataset(filename, dict_dataset):
    with h5py.File(filename, 'w') as f:
        for k, v in dict_dataset.items():
            f.create_dataset(k, data=v)


def hz_to_mel(hz):
    mel = (2595 * np.log10(1 + hz / 700))  # Convert Hz to Mel
    return mel


def mel_to_hz(mel):
    hz = (700 * (10 ** (mel / 2595) - 1))  # Convert Mel to Hz
    return hz


def mel_space_dafe(freq_low, freq_high, n_channels):
    low_freq_mel = hz_to_mel(freq_low)
    high_freq_mel = hz_to_mel(freq_high)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_channels + 2)  # Equally spaced in Mel scale
    # space_mel = mel_points[1] - mel_points[0]
    # mel_points = np.insert(mel_points, 0, mel_points[0] - space_mel)
    # mel_points = np.append(mel_points, mel_points[-1] + space_mel)
    hz_points = mel_to_hz(mel_points)  # Convert Mel to Hz
    return hz_points


def log_space_dafe(freq_low, freq_high, n_channels):
    low_freq_log = np.log10(freq_low)
    high_freq_log = np.log10(freq_high)
    hz_points = np.logspace(low_freq_log, high_freq_log, n_channels)
    space_log = (high_freq_log - low_freq_log) / float(n_channels - 1)
    hz_points = np.insert(hz_points, 0, 10 ** (np.log10(hz_points[0]) - space_log))
    hz_points = np.append(hz_points, 10 ** (np.log10(hz_points[-1]) + space_log))
    return hz_points


def mel_space_aafe(freq_low, freq_high, n_channels):
    low_freq_mel = hz_to_mel(freq_low)
    high_freq_mel = hz_to_mel(freq_high)

    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_channels)  # Equally spaced in Mel scale
    hz_points = mel_to_hz(mel_points)  # Convert Mel to Hz
    return hz_points


def log_space_aafe(freq_low, freq_high, n_channels):
    low_freq_log = np.log10(freq_low)
    high_freq_log = np.log10(freq_high)
    hz_points = np.logspace(low_freq_log, high_freq_log, n_channels)
    return hz_points


def get_dafe_filters(args, config, df_description, plot=0):
    fbank = None
    filt_b = None
    filt_a = None
    freq_low = config['feature']['freq_low']
    freq_high = config['feature']['freq_high']
    n_filt = config['feature']['n_filt']
    spacing = config['feature']['dafe']['f_spacing']
    oversample_factor = config['feature']['oversample_factor']
    nfft = config['feature']['dafe']['nfft']
    # Get Sample Rate
    for row in df_description.itertuples():
        filepath = row.path
        if platform.system() == 'Windows':
            filepath.replace("/", "\\")
        import soundfile as snd
        # sample_rate, _ = wavfile.read(filepath)
        # filepath = filepath.replace("C:\\", "/")
        filepath = filepath.replace("\\", "/")
        filepath = filepath.replace("C:/", "/")

        _, sample_rate = snd.read(filepath, dtype='int16')
        sample_rate = float(sample_rate)
        break

    if oversample_factor > 1:
        sample_rate *= oversample_factor

    # Nyquist Frequency
    nyq = sample_rate / 2.0

    # Set Upper Frequency
    if freq_high is None:
        freq_high = sample_rate / 2 - 500

    # Get Spacing Points
    ###################################
    # Build TRI
    ###################################

    # Filter Banks
    if spacing == 'mel':
        hz_points = mel_space_dafe(freq_low, freq_high, n_filt)
    elif spacing == 'log':
        hz_points = log_space_dafe(freq_low, freq_high, n_filt)

    # Create Filters
    # Limit Hz Point between [0, sample_rate/2]
    hz_points = np.clip(hz_points, 0, sample_rate / 2)
    fbank_bin = np.floor((nfft + 1) * hz_points / sample_rate)
    fbank = np.zeros((n_filt, int(np.floor(nfft / 2 + 1))))

    # Plot
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        freq_step = sample_rate / 2 / (nfft / 2)
        freq = np.arange(0, int(nfft / 2) + 1) * freq_step

    for m in range(1, n_filt + 1):
        f_m_minus = int(fbank_bin[m - 1])  # left
        f_m = int(fbank_bin[m])  # center
        f_m_plus = int(fbank_bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - fbank_bin[m - 1]) / (fbank_bin[m] - fbank_bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (fbank_bin[m + 1] - k) / (fbank_bin[m + 1] - fbank_bin[m])

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

    print("Hz Point: ", hz_points)
    return fbank, filt_b, filt_a, hz_points


def get_aafe_filters(args, config, df_description, plot=0):
    fbank = None
    oversample_factor = config['feature']['oversample_factor']
    freq_low = config['feature']['freq_low']
    freq_high = config['feature']['freq_high']
    n_filt = config['feature']['n_filt']
    sample_rate = 0
    q_factors = np.asarray(config['feature']['aafe']['q_factors'])
    ch_orders = np.asarray(config['feature']['aafe']['ch_orders'])
    # Get Sample Rate
    for row in df_description.itertuples():
        filepath = row.path
        if platform.system() == 'Windows':
            filepath.replace("/", "\\")
        filepath = filepath.replace("\\", "/")
        filepath = filepath.replace("C:/", "/")
        import soundfile as snd
        # sample_rate, _ = wavfile.read(filepath)
        _, sample_rate = snd.read(filepath, dtype='int16')
        sample_rate = float(sample_rate)
        break

    if oversample_factor > 1:
        sample_rate *= oversample_factor

    # Nyquist Frequency
    nyq = sample_rate / 2.0

    # Set Upper Frequency
    if freq_high is None:
        freq_high = sample_rate / 2 - 500

    ###################################
    # Build BPF
    ###################################

    # Generate the array of center frequencies for band-pass filters
    if config['feature']['aafe']['f_spacing'] == 'mel':
        hz_points = mel_space_aafe(freq_low, freq_high, n_filt)
    elif config['feature']['aafe']['f_spacing'] == 'log':
        hz_points = log_space_aafe(freq_low, freq_high, n_filt)
    else:
        hz_points = np.asarray(config['feature']['aafe']['f_center'])

    # Generate the array of bandwidth for band-pass filters
    bw_array = hz_points / q_factors

    # Generate the filter coefficients
    filt_b = []
    filt_a = []
    for i in range(config['feature']['n_filt']):
        b, a = signal.butter(N=ch_orders[i],  # The order of the filter
                             Wn=[(hz_points[i] - (bw_array[i]) / 2) / nyq,  # Corner frequencies for BPF.
                                 (hz_points[i] + (bw_array[i]) / 2) / nyq],  # Normalized to [0, 1], where 1 is nyq
                             btype='band',  # Bandpass filter
                             output='ba')  # Numerator (b) and denominator (a) polynomials of the IIR filter.
        filt_b.append(b)
        filt_a.append(a)

    if plot:
        # Generate the filter response
        filt_w = []
        filt_h = []
        for i in range(n_filt):
            w, h = signal.freqz(filt_b[i], filt_a[i], worN=163840)
            w = w[1:]  # Exclude DC component
            h = h[1:]  # Exclude DC component
            filt_w.append(w)
            filt_h.append(h)

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        for i in range(n_filt):
            axes[0].semilogx((filt_w[i] / (2 * np.pi)) * sample_rate, 20 * np.log10(abs(filt_h[i])))
            axes[0].set_xlim([1e0, 1e6])
            axes[0].set_xlabel('log(Frequency)', fontsize=18)
            axes[0].set_ylabel('dB', fontsize=18)
            axes[0].tick_params(labelsize=16)
            axes[1].plot((filt_w[i] / (2 * np.pi)) * sample_rate, abs(filt_h[i]))
            axes[1].set_xlabel('Frequency', fontsize=18)
            axes[1].set_ylabel('Amplitude Gain', fontsize=18)
            axes[1].tick_params(labelsize=16)
            axes[1].set_xlim(0, 16000)
        axes[0].grid(which='both', axis='both')
        axes[0].set_ylim(bottom=-100)
        axes[1].grid(which='both', axis='both')
        # plt.semilogx((filt_w[i] / (2 * np.pi)) * sample_rate, 20 * np.log10(abs(filt_h[i])))

        # plt.grid(which='both', axis='both')
        # plt.xlim(left=10)
        # plt.ylim(bottom=-100)
        plt.show()
    print("Hz Point: ", hz_points)
    return fbank, filt_b, filt_a, hz_points


def compute_fbank(signal, sample_rate, frame_size=0.025, frame_stride=0.01, NFFT=512, nfilt=40):
    # Initialization
    np.random.seed(2)

    sample_rate = float(sample_rate)

    # Framing
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal,
                           z)  # Pad Signal to make sure that all frames have equal number of samples without
    # truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    frames = pad_signal[indices.astype(np.int32, copy=False)]

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
        if (std != 0):
            batch_curr /= std
        norm_frames = np.vstack((norm_frames, batch_curr))

    frames = norm_frames

    # Windowing
    frames *= np.hamming(frame_length)

    # Fourier-Transform and Power Spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    # Filter Banks
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = np.log(filter_banks)  # dB
    features = filter_banks

    return features

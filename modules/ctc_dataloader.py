import numpy as np
import h5py

class CTCDataLoader(object):
    def __init__(self, trainfile, devfile, testfile):
        self.mean_train = 0
        self.std_train = 0
        self.trainfile = trainfile
        self.devfile = devfile
        self.testfile = testfile
        self.mean_train = 0
        self.std_train = 0

        with h5py.File(self.trainfile) as hf:
            feats = hf.get('features')
            self.mean_train = np.mean(feats[:, :], axis=0)
            self.std_train = np.std(feats[:, :], axis=0)

    def iterate(self,
                epoch,
                h5,
                cache_size=1000,
                batch_size=32,
                mode='batch',
                shuffle_type='high_throughput',
                normalization=0,
                enable_gauss=0,
                ctc_mode='first'):

        np.random.seed(epoch)

        if ctc_mode == 'first':
            ctc_label_shift = 1
        else:
            ctc_label_shift = 0

        with h5py.File(h5) as hf:

            feats = hf.get('features')
            feature_lens = np.array(hf.get('feature_lens')).astype(int)
            labels = np.array(hf.get('labels')).astype(int)
            label_lens = np.array(hf.get('label_lens')).astype(int)

            feat_slice_idx = idx_to_slice(feature_lens)
            label_slice_idx = idx_to_slice(label_lens)

            # Constants
            n_samples = len(feature_lens)
            ndim = len(feats[1, :])

            # Batch Shuffle
            if shuffle_type == 'high_throughput':
                s_idx = np.argsort(feature_lens)[::-1]
            elif shuffle_type == 'random':
                s_idx = np.random.permutation(n_samples)
            else:
                s_idx = range(n_samples)
            batches_idx = create_batch_idx(s_idx=s_idx,
                                           feature_lens=feature_lens,
                                           cache_size=cache_size,
                                           batch_size=batch_size,
                                           mode=mode)

            n_batches = len(batches_idx)  # Number of batches
            b = 0
            while b < n_batches:
                curr_batch_idx = batches_idx[b]

                # Load batch
                batch_feats = []
                batch_labels = []
                for sample_idx in curr_batch_idx:
                    batch_feats.append(feats[slc(sample_idx, feat_slice_idx), :])
                    batch_labels.append(labels[slc(sample_idx, label_slice_idx)])

                # Normalize batch
                if normalization:
                    batch_feats = normalize(batch_feats, self.mean_train, self.std_train)

                # Add gaussian noise:
                if enable_gauss != 0.0:
                    batch_feats = add_gaussian_noise(batch_feats, sigma=enable_gauss)

                # Zero Padding
                max_len = np.max(feature_lens[curr_batch_idx])
                bX = np.zeros((len(curr_batch_idx), max_len, ndim), dtype='float32')
                bY = []
                b_lenY = np.zeros((len(curr_batch_idx)), dtype='int32')
                b_lenX = feature_lens[curr_batch_idx].astype('int32')

                for i, sample in enumerate(batch_feats):
                    len_sample = sample.shape[0]
                    bX[i, :len_sample, :] = sample

                    ctc_labels = np.asarray(batch_labels[i]) + ctc_label_shift  # Make label 0 the 'blank'
                    bY.extend(ctc_labels)
                    b_lenY[i] = len(ctc_labels)
                bY = np.asarray(bY, dtype='int32')
                b += 1

                yield bX, b_lenX, bY, b_lenY


def idx_to_slice(lens):
    idx = []
    lens_cs = np.cumsum(lens)
    for i, len in enumerate(lens):
        idx.append((lens_cs[i] - lens[i], lens_cs[i]))
    return idx


def slc(i, idx):
    return slice(idx[i][0], idx[i][1])


def normalize(batch, ep_mean, ep_std):
    batch_normalized = []
    for sample in batch:
        sample = sample - ep_mean[np.newaxis, :]
        sample = sample / ep_std[np.newaxis, :]
        batch_normalized.append(sample)

    return batch_normalized


def add_gaussian_noise(batch_feats, sigma=0.6):

    batch_gauss = []
    for sample in batch_feats:
        noise_mat = sigma * np.random.standard_normal(sample.shape)
        sample = sample + noise_mat

        batch_gauss.append(sample)
    return batch_gauss


def create_batch_idx(s_idx, feature_lens, cache_size, batch_size, mode='batch'):
    list_batches = []
    batch = []
    max_len = 0

    if mode == 'batch':
        for i, sample in enumerate(s_idx):
            if len(batch) < batch_size:
                batch.append(sample)
            else:
                list_batches.append(batch)
                batch = [sample]
    elif mode == 'cache':
        for i, sample in enumerate(s_idx):
            max_len = max(feature_lens[sample], max_len)
            num_frames = (len(batch) + 1) * max_len
            if num_frames <= cache_size:
                batch.append(sample)
            else:
                list_batches.append(batch)
                batch = [sample]
                max_len = feature_lens[sample]

    list_batches.append(batch)
    return list_batches

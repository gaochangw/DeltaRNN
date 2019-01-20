import numpy as np
import h5py


class DataLoader(object):
    def __init__(self, trainfile):
        self.mean_train = 0
        self.std_train = 0
        self.trainfile = trainfile
        self.mean_train = 0
        self.std_train = 0

        with h5py.File(self.trainfile) as hf:
            feats = np.asarray(hf.get('features'), dtype=np.float32)
            labels = np.asarray(hf.get('labels'), dtype=int)
            self.n_samples = labels.shape[0]
            self.n_feature = feats.shape[1]
            self.mean_train = np.mean(feats[:, :], axis=0)
            self.std_train = np.std(feats[:, :], axis=0)
            # unique, counts = np.unique(labels, return_counts=True)
            # self.n_impostor = counts[0]
            # self.n_client = counts[1]

    def iterate(self,
                epoch,
                h5,
                batch_size=32,
                shuffle_type='random',
                normalization=1,
                enable_gauss=0):

        np.random.seed(epoch)

        with h5py.File(h5) as hf:

            feats = np.asarray(hf.get('features'), dtype=np.float32)
            labels = np.asarray(hf.get('labels'), dtype=int)

            batches_idx = create_batch_idx(n_samples=self.n_samples,
                                           batch_size=batch_size,
                                           shuffle_type=shuffle_type)

            n_batches = len(batches_idx)  # Number of batches
            #print(n_batches)
            b = 0
            while b < n_batches:
                curr_batch_idx = batches_idx[b]

                # Load batch
                batch_feats = []
                batch_labels = []
                for sample_idx in curr_batch_idx:
                    batch_feats.append(feats[sample_idx, :])
                    batch_labels.append(labels[sample_idx])

                bX = np.stack(batch_feats, axis=0)
                bY = np.stack(batch_labels, axis=0)

                # Normalize batch
                if normalization:
                    batch_feats = normalize(bX, self.mean_train, self.std_train)

                # Add gaussian noise:
                if enable_gauss != 0.0:
                    batch_feats = add_gaussian_noise(bX, sigma=enable_gauss)

                b += 1

                yield bX, bY


def idx_to_slice(lens):
    idx = []
    lens_cs = np.cumsum(lens)
    for i, len in enumerate(lens):
        idx.append((lens_cs[i] - lens[i], lens_cs[i]))
    return idx


def slc(i, idx):
    return slice(idx[i][0], idx[i][1])


def normalize(batch, ep_mean, ep_std):
    batch = batch - ep_mean[np.newaxis, :]
    batch = batch / ep_std[np.newaxis, :]
    return batch


def add_gaussian_noise(batch_feats, sigma=0.6):

    batch_gauss = []
    for sample in batch_feats:
        noise_mat = sigma * np.random.standard_normal(sample.shape)
        sample = sample + noise_mat

        batch_gauss.append(sample)
    return batch_gauss


def create_batch_idx(n_samples, batch_size, shuffle_type='random'):
    list_batches = []
    batch = []

    # Batch Shuffle
    if shuffle_type == 'random':
        s_idx = np.random.permutation(n_samples)
    else:
        s_idx = range(n_samples)

    for sample in s_idx:
        if len(batch) < batch_size:
            batch.append(sample)
        else:
            list_batches.append(batch)
            batch = [sample]

    list_batches.append(batch)
    return list_batches

if __name__ == "__main__":

    h5 = '../feat/train_vad.h5'

    with h5py.File(h5) as hf:

        feats = np.asarray(hf.get('features'), dtype=np.float32)
        labels = np.asarray(hf.get('labels'), dtype=int)

        # Constants
        n_samples = labels.shape[0]
        n_feature = feats.shape[1]

        batches_idx = create_batch_idx(n_samples=n_samples,
                                        batch_size=32,
                                        shuffle_type='random')

        n_batches = len(batches_idx)  # Number of batches
        b = 0
        while b < n_batches:
            curr_batch_idx = batches_idx[b]

            # Load batch
            batch_feats = []
            batch_labels = []
            for sample_idx in curr_batch_idx:
                batch_feats.append(feats[sample_idx, :])
                batch_labels.append(labels[sample_idx])

            batch_feats = np.stack(batch_feats, axis=0)
            batch_labels = np.stack(batch_labels, axis=0)
    
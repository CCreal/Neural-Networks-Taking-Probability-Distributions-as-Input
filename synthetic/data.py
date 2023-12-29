import numpy as np
from torch.utils import data

class SynData(data.Dataset):
    def __init__(self, number_datasets, sample_size, number_features, distributions):
        self.number_datasets = number_datasets
        self.sample_size = sample_size
        self.number_features = number_features
        self.distributions = distributions

        self.label_dict = {distributions[i]:i for i in range(len(distributions))}
        print(self.label_dict)

        self.data = self.create_sets()
        self.datasets = self.data['datasets']
        self.labels = self.data['labels']
        self.labels_index = self.data['labels_index']

    def __getitem__(self, item):

        return (self.datasets[item], self.labels_index[item])

    def __len__(self):
        return self.number_datasets

    def generate_distribution(self, distribution):
        m = np.random.uniform(-1, 1)
        v = np.random.uniform(0.5, 2)

        if distribution == 'gaussian':
            samples = np.random.normal(m, v, (self.sample_size, self.number_features))
            return samples, m, v

        elif distribution == 'mixture of gaussians':
            mix_1 = np.random.normal(-(1 + np.abs(m)), v / 2, (int(self.sample_size / 2),
                                                               self.number_features))
            mix_2 = np.random.normal((1 + np.abs(m)), v / 2, (int(self.sample_size / 2),
                                                              self.number_features))
            return np.vstack((mix_1, mix_2)), 1 + np.abs(m), v / 2

        elif distribution == 'exponential':
            samples = np.random.exponential(1, (self.sample_size, self.number_features))
            return self.augment_distribution(samples, m, v), m, v

        elif distribution == 'reverse exponential':
            samples = - np.random.exponential(1, (self.sample_size, self.number_features))
            return self.augment_distribution(samples, m, v), m, v

        elif distribution == 'laplacian':
            samples = np.random.laplace(m, v, (self.sample_size, self.number_features))
            return samples, m, v

        elif distribution == 'uniform':
            samples = np.random.uniform(-1, 1, (self.sample_size, self.number_features))
            return self.augment_distribution(samples, m, v), m, v

        elif distribution == 'negative binomial':
            samples = np.random.negative_binomial(50, 0.5, (self.sample_size,
                                                            self.number_features))
            samples = np.asarray(samples, dtype=np.float64)
            return self.augment_distribution(samples, m, v), m, v

        else:
            print("Unrecognised choice of distribution.")
            return None

    @staticmethod
    def augment_distribution(samples, m, v):
        aug_samples = samples.copy()
        aug_samples -= np.mean(samples)
        aug_samples /= np.std(samples)
        aug_samples *= v ** 0.5
        aug_samples += m
        return aug_samples

    def create_sets(self):
        sets = np.zeros((self.number_datasets, self.sample_size, self.number_features),
                        dtype=np.float32)
        labels = []
        labels_index = []
        means = []
        variances = []

        for i in range(self.number_datasets):
            distribution = np.random.choice(self.distributions)

            x, m, v = self.generate_distribution(distribution)

            sets[i, :, :] = x
            labels.append(distribution)
            labels_index.append(self.label_dict[distribution])
            means.append(m)
            variances.append(v)

        return {
            "datasets": sets,
            "labels": np.array(labels),
            'labels_index': np.array(labels_index),
            "means": np.array(means),
            "variances": np.array(variances),
            "distributions": np.array(self.distributions)
        }
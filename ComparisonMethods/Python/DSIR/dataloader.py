from __future__ import print_function
import torch
import torch.utils.data as data
import numpy as np


class MultiOmicsDataset(data.Dataset):
    def __init__(self, data_list, transform=None):
        """
        Initialize dataset with a list of data arrays

        Parameters:
        -----------
        data_list : list of numpy arrays
            Each array should be (n_samples, n_features)
        transform : callable, optional
            Optional transform to be applied on a sample
        """
        self.transform = transform
        self.data_list = []

        # Validate all data have same number of samples
        n_samples = data_list[0].shape[0]
        for i, data in enumerate(data_list):
            if data.shape[0] != n_samples:
                raise ValueError(f"All data must have same number of samples. "
                                 f"Data {i} has {data.shape[0]} samples, expected {n_samples}")

            # Convert to float32 and store
            self.data_list.append(data.astype(np.float32))

        self.n_samples = n_samples
        self.n_modalities = len(data_list)

        print(f"Dataset initialized with {self.n_modalities} modalities:")
        for i, data in enumerate(self.data_list):
            print(f"  Modality {i}: {data.shape}")

    def __getitem__(self, index):
        """Return samples from all modalities for given index"""
        samples = tuple(data[index, :] for data in self.data_list)
        return samples

    def __len__(self):
        return self.n_samples


def create_dataloader(data_list, batch_size=None, shuffle=False):
    """
    Create DataLoader from list of data arrays

    Parameters:
    -----------
    data_list : list of numpy arrays
        Each array should be (n_samples, n_features)
    batch_size : int, optional
        If None, uses all samples as one batch
    shuffle : bool, optional
        Whether to shuffle data

    Returns:
    --------
    DataLoader object
    """
    dataset = MultiOmicsDataset(data_list)

    if batch_size is None:
        batch_size = dataset.n_samples

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
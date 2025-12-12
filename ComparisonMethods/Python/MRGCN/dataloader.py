from __future__ import print_function
import torch
import torch.utils.data as data
import numpy as np


def load_data_from_array(gene_expression):
    """
    Process gene expression data that's already loaded (from R)
    Args:
        gene_expression: numpy array of gene expression data
    """
    Label = gene_expression
    # Reshape similar to original load_data function
    Img = np.reshape(gene_expression, [gene_expression.shape[0], 1, gene_expression.shape[1], 1])
    n_input = [1, gene_expression.shape[1]]
    return gene_expression, Img, Label, n_input


class EYB_FromArrays(data.Dataset):
    def __init__(self, *data_arrays, transform=None):
        """
        Initialize dataset with variable number of pre-loaded data arrays
        Args:
            *data_arrays: Variable number of numpy arrays containing expression data
            transform: optional transform to apply
        """
        self.transform = transform
        self.n_modalities = len(data_arrays)

        # Process each input data array
        self.processed_data = []

        for i, data_array in enumerate(data_arrays):
            # Ensure proper data type handling
            data_array = np.array(data_array, dtype=np.float32)
            # Handle any NaN or infinite values
            data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)

            gene_expression, Img, Label, n_input = load_data_from_array(data_array)

            # Apply specific processing for the third modality (if it exists) - keeping original logic
            # if i == 2 and Img.shape[2] > 885:
            #     Img = Img[:, :, 0:885, :]

            # Squeeze dimensions
            Img = np.squeeze(Img, axis=None)
            Img = Img.astype(np.float32)

            self.processed_data.append(Img)

        # Set train_num based on the first modality's size (assuming all have same number of samples)
        self.train_num = self.processed_data[0].shape[0]

        # Print shapes for debugging
        shapes_str = ", ".join([str(data.shape) for data in self.processed_data])
        print(f"Data shapes for {self.n_modalities} modalities: {shapes_str}")

    def __getitem__(self, index):
        # Return data from all modalities for the given index
        return tuple(data[index, :] for data in self.processed_data)

    def __len__(self):
        return self.train_num


def create_data_loader_from_arrays(*data_arrays, batch_size=None, shuffle=False):
    """
    Create a data loader from variable number of numpy arrays
    Args:
        *data_arrays: Variable number of numpy arrays
        batch_size: batch size (if None, uses the full dataset size)
        shuffle: whether to shuffle data
    """
    dataset = EYB_FromArrays(*data_arrays)
    if batch_size is None:
        batch_size = len(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader
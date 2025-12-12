import sys

import pandas as pd

sys.path.append("/data/daotran/Cancer_RP/Subtyping/BiB_Subtyping/GitFolder/Python/DSIR")

import torch
import numpy as np
from dataloader import create_dataloader
from network import FlexibleMultiOmicsNetwork
from snf_sm import create_snf_network
import random


def set_seed(seed=1234):
    """Set seeds for reproducible results"""
    # Convert seed to integer to handle R->Python conversion
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_coef_matrix(data_list,
                        n_epochs1=50,
                        n_epochs2=200,
                        learning_rate=0.001,
                        latent_dim=15,
                        kernel_size=5,
                        snf_K=20,
                        snf_mu=0.5,
                        snf_t=20,
                        loss_weights=None,
                        device_name="cpu",
                        verbose=True,
                        seed=1234):
    """
    Main function to compute coefficient matrix from multi-omics data

    Parameters:
    -----------
    data_list : list of numpy arrays
        Each array should be (n_samples, n_features)
    n_epochs1 : int, optional
        Number of epochs for initial autoencoder training (default: 50)
    n_epochs2 : int, optional
        Number of epochs for SNF-guided training (default: 200)
    learning_rate : float, optional
        Learning rate for optimizer (default: 0.001)
    latent_dim : int, optional
        Latent dimension for encoders (default: 15)
    kernel_size : int, optional
        Convolution kernel size (default: 5)
    snf_K : int, optional
        Number of neighbors for SNF (default: 20)
    snf_mu : float, optional
        SNF scaling parameter (default: 0.5)
    snf_t : int, optional
        Number of SNF iterations (default: 20)
    loss_weights : dict, optional
        Weights for different loss components. Keys: 'reconstruction', 'encoding', 'regularization', 'fusion'
        Default: {'reconstruction': 1.0, 'encoding': 1.0, 'regularization': 10.0, 'fusion': -20.0}
    device_name : str, optional
        Device to use ('cpu' or 'cuda') (default: 'cpu')
    verbose : bool, optional
        Whether to print progress (default: True)

    Returns:
    --------
    coef_matrix : numpy array
        Final coefficient matrix (n_samples, n_samples)
    """

    # Set device

    seed = int(seed)
    set_seed(seed)

    device = torch.device(device_name)

    # Default loss weights
    if loss_weights is None:
        loss_weights = {
            'reconstruction': 1.0,
            'encoding': 1.0,
            'regularization': 10.0,
            'fusion': -20.0
        }

    # Convert data_list to numpy arrays if needed
    data_arrays = []
    for i, data in enumerate(data_list):
        np_data = np.array(data, dtype=np.float32)
        # Handle any NaN or infinite values
        np_data = np.nan_to_num(np_data, nan=0.0, posinf=0.0, neginf=0.0)
        data_arrays.append(np_data)

    # After data validation, before creating the network
    for i, data in enumerate(data_arrays):
        current_features = data.shape[1]
        remainder = current_features % kernel_size
        if remainder != 0:
            padding_needed = kernel_size - remainder
            # Pad with zeros
            padding = np.zeros((data.shape[0], padding_needed))
            data_arrays[i] = np.concatenate([data, padding], axis=1)
            print(f"Padded modality {i} from {current_features} to {data_arrays[i].shape[1]} features")

    # Get dimensions
    n_samples = data_arrays[0].shape[0]
    feature_dims = [data.shape[1] for data in data_arrays]
    n_modalities = len(data_arrays)

    # return(n_samples + n_modalities)

    if verbose:
        print(f"Dataset info:")
        print(f"  Number of samples: {n_samples}")
        print(f"  Number of modalities: {n_modalities}")
        print(f"  Feature dimensions: {feature_dims}")

    # Validate feature dimensions are divisible by kernel_size
    for i, feat_dim in enumerate(feature_dims):
        if feat_dim % kernel_size != 0:
            raise ValueError(f"Feature dimension {feat_dim} for modality {i} "
                             f"must be divisible by kernel_size {kernel_size}")

    # Create dataloader
    dataloader = create_dataloader(data_arrays, batch_size=n_samples, shuffle=False)

    # Initialize model
    model = FlexibleMultiOmicsNetwork(
        feature_dims=feature_dims,
        n_samples=n_samples,
        latent_dim=latent_dim,
        kernel_size=kernel_size
    ).to(device)

    # Loss functions and optimizer
    mse_criterion = torch.nn.MSELoss(reduction='sum')
    l1_criterion = torch.nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.0
    )

    if verbose:
        print(f"\nPhase 1: Initial autoencoder training ({n_epochs1} epochs)")
        print("-" * 50)

    # Phase 1: Initial autoencoder training
    model.train()
    for epoch in range(n_epochs1):
        for batch_data in dataloader:
            # Reshape inputs to (batch_size, 1, features, 1)
            inputs = []
            for i, data in enumerate(batch_data):
                input_tensor = data.view(n_samples, 1, feature_dims[i], 1).to(device)
                inputs.append(input_tensor)

            # Forward pass
            outputs = model(inputs)

            # Compute reconstruction loss
            loss = sum(mse_criterion(output, input_tensor)
                       for output, input_tensor in zip(outputs, inputs))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}/{n_epochs1}, Loss: {loss.item():.4f}")

    # Create SNF network
    if verbose:
        print(f"\nCreating SNF network...")

    fused_network = create_snf_network(
        data_arrays,
        metric='euclidean',
        K=snf_K,
        mu=snf_mu,
        t=snf_t
    )
    fused_network_tensor = torch.tensor(fused_network, dtype=torch.float32).to(device)

    if verbose:
        print(f"SNF network created with shape: {fused_network.shape}")
        print(f"\nPhase 2: SNF-guided training ({n_epochs2} epochs)")
        print("-" * 50)

    # Phase 2: SNF-guided training
    optimizer2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.0
    )

    for epoch in range(n_epochs2):
        for batch_data in dataloader:
            # Reshape inputs
            inputs = []
            for i, data in enumerate(batch_data):
                input_tensor = data.view(n_samples, 1, feature_dims[i], 1).to(device)
                inputs.append(input_tensor)

            # Forward pass with relationships
            z_list, outputs, zcoef_list, coef = model.forward2(inputs)

            # Compute losses
            # Reconstruction loss
            loss_r = sum(mse_criterion(output, input_tensor)
                         for output, input_tensor in zip(outputs, inputs))

            # Encoding consistency loss
            loss_e = sum(mse_criterion(zcoef, z)
                         for zcoef, z in zip(zcoef_list, z_list))

            # Regularization loss (sparsity)
            loss_re = mse_criterion(coef, torch.zeros_like(coef))

            # Fusion guidance loss
            loss_ff = l1_criterion(
                coef.mul(fused_network_tensor),
                torch.zeros_like(coef)
            )

            # Combined loss
            loss = (loss_weights['reconstruction'] * loss_r +
                    loss_weights['encoding'] * loss_e +
                    loss_weights['regularization'] * loss_re +
                    loss_weights['fusion'] * loss_ff)

            # Backward pass
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()

        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch}/{n_epochs2}, Loss: {loss.item():.4f}")
            print(f"  Reconstruction: {loss_r.item():.4f}")
            print(f"  Encoding: {loss_e.item():.4f}")
            print(f"  Regularization: {loss_re.item():.4f}")
            print(f"  Fusion: {loss_ff.item():.4f}")

    # Extract final coefficient matrix
    with torch.no_grad():
        final_coef = (model.weight - torch.diag(torch.diag(model.weight))).cpu().numpy()

    # Save to file if path provided
    if verbose:
        print(f"\nTraining completed!")
        print(f"Final coefficient matrix shape: {final_coef.shape}")

    return final_coef


# import pandas as pd
#
# mRNA = pd.read_csv(
#     "/data/daotran/Cancer_RP/Subtyping/BiB_Subtyping/NewMethods/Deep-Latent-Space-Fusion/aligned_exp.csv", index_col=0)
# mRNA = mRNA.T
#
# miRNA = pd.read_csv(
#     "/data/daotran/Cancer_RP/Subtyping/BiB_Subtyping/NewMethods/Deep-Latent-Space-Fusion/aligned_mirna.csv",
#     index_col=0).T
# data_list = [mRNA, miRNA]
#
# tss = compute_coef_matrix(data_list)

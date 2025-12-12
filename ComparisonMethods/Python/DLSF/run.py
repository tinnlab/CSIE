import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import cluster
from scipy.spatial import distance
from sklearn.utils.validation import check_array
import time
import tempfile
import random

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CycleAE(nn.Module):
    def __init__(self, input_dims):
        super(CycleAE, self).__init__()

        assert isinstance(input_dims, list)
        self.encoder = nn.Sequential(
            nn.Linear(input_dims[0], input_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(input_dims[1], input_dims[2]),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dims[2], input_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(input_dims[1], input_dims[0]),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        z_recon = self.encoder(x_recon)
        return z, x_recon, z_recon


def train_CycleAE(model, X, epochs=1000, lr=1e-3, weight_xx=1.0, weight_zz=1.0, device='cpu', show_freq=-1):
    # ensure input in the form of Tensor
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(np.array(X), dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_record = []
    for epoch in range(epochs):
        Z, X_recon, Z_recon = model.forward(X)
        loss1 = F.mse_loss(X_recon, X, reduction='mean')
        loss2 = F.mse_loss(Z_recon, Z, reduction='mean')
        loss = weight_xx * loss1 + weight_zz * loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_record.append([loss1, loss2])
        if (epoch % show_freq == 0 or epoch == epochs - 1) and (show_freq != -1):
            # loss: X-X, Z-Z
            print('Epoch {}: Loss_XX: {}'.format(epoch, loss1))
            print('Epoch {}: Loss_ZZ: {}'.format(epoch, loss2))
            print('Epoch {}: total loss: {}'.format(epoch, loss))

    return loss_record


class SelfExpression(nn.Module):
    def __init__(self, n, knn):
        super(SelfExpression, self).__init__()
        self.knn = knn
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones((n, n), dtype=torch.float32), requires_grad=True)

    def forward(self, z):  # shape=[n, d]
        coef = self.Coefficient
        coef = coef * torch.tensor(self.knn > 0, dtype=torch.float32)
        coef = coef * (coef > 0)
        self.Coefficient.data = coef
        zs = torch.matmul(coef, z)  # z self-expression
        return zs


class DMSCNet(nn.Module):
    def __init__(self, num_omics, input_dims, num_samples, kernel, w):
        super(DMSCNet, self).__init__()
        self.K = num_omics
        self.n = num_samples
        self.knn = kernel
        self.w = w
        self.CAE = [CycleAE(input_dims[k]) for k in range(num_omics)]
        self.self_expression = SelfExpression(self.n, self.knn)

    def forward(self, X):  # shape=[n, d]
        Z = []
        Z_selfExp = []
        X_recon = []
        Z_recon = []
        for k in range(self.K):
            x_input = X[k]
            z = self.CAE[k].encoder(x_input)
            z_selfExp = self.self_expression(z)
            x_recon = self.CAE[k].decoder(z_selfExp)
            z_recon = self.CAE[k].encoder(x_recon)

            Z.append(z)
            Z_selfExp.append(z_selfExp)
            X_recon.append(x_recon)
            Z_recon.append(z_recon)

        return Z, Z_selfExp, X_recon, Z_recon

    def loss_fn(self, X, X_recon, Z, Z_selfExp, Z_recon, weight_xx, weight_zz, weight_selfExp, weight_coef):
        Loss_XX = []
        Loss_ZZ = []
        Loss_SelfExp = []
        for k in range(self.K):
            x, x_recon, z, z_selfExp, z_recon = X[k], X_recon[k], Z[k], Z_selfExp[k], Z_recon[k]
            loss_xx = F.mse_loss(x_recon, x, reduction='mean')
            loss_zz = F.mse_loss(z_recon, z, reduction='mean')
            loss_selfExp = F.mse_loss(z_selfExp, z, reduction='mean') * self.w[k]
            Loss_XX.append(loss_xx)
            Loss_ZZ.append(loss_zz)
            Loss_SelfExp.append(loss_selfExp)

        loss_coef = F.mse_loss(torch.sum(self.self_expression.Coefficient, dim=1), torch.ones(self.n), reduction='mean')
        total_loss = weight_xx * sum(Loss_XX) + weight_zz * sum(Loss_ZZ) + weight_selfExp * sum(Loss_SelfExp) \
                     + weight_coef * loss_coef

        return total_loss, [Loss_XX, Loss_ZZ, Loss_SelfExp, loss_coef]


def train_DMSCNet(model, X, epochs=100, num_omics=None, lr=1e-3, weight_xx=1.0, weight_zz=0.2,
                  weight_selfExp=1.0, weight_coef=1.0, device='cpu', show_freq=-1):
    if num_omics is None:
        num_omics = len(X)

    for k in range(num_omics):
        if not isinstance(X[k], torch.Tensor):
            X[k] = torch.tensor(X[k], dtype=torch.float32, device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    losses_item = []

    for epoch in range(epochs):
        Z, Z_selfExp, X_recon, Z_recon = model(X)
        loss, loss_item = model.loss_fn(X, X_recon, Z, Z_selfExp, Z_recon, weight_xx, weight_zz, weight_selfExp,
                                        weight_coef)
        losses.append(loss)
        losses_item.append(loss_item)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch % show_freq == 0 or epoch == epochs - 1) and (show_freq != -1):
            print('Epoch {}: total loss: {}'.format(epoch, loss))

    return losses, losses_item


def keep_high_var_features(omics_list, num_features=2000):
    retained_omics_list = []
    for i in range(len(omics_list)):
        temp_omics = omics_list[i]
        if temp_omics.shape[1] > num_features:
            features_vars = temp_omics.var(axis=0)
            threshold = sorted(features_vars, reverse=True)[num_features]
            new_omics = temp_omics.loc[:, features_vars > threshold]
            retained_omics_list.append(new_omics)
        else:
            retained_omics_list.append(temp_omics)

    return retained_omics_list


def normalize_matrix(omics_list, type='min-max'):
    retained_omics_list = []
    for i in range(len(omics_list)):
        temp_omics = omics_list[i]
        if type == 'z-score':
            new_omics = preprocessing.scale(temp_omics, axis=0)
            retained_omics_list.append(new_omics)
        elif type == 'min-max':
            new_omics = preprocessing.minmax_scale(temp_omics, axis=0)
            retained_omics_list.append(new_omics)
        else:
            print("Error! required z-score or min-max")

    return retained_omics_list


def knn_kernel(data, k=10):
    # knn sample weight set to 1 for each row
    num_samples = data.shape[0]
    Dis_Mat = distance.cdist(data, data)
    kernel = np.ones_like(Dis_Mat)
    sort_dist = np.sort(Dis_Mat, axis=1)
    threshold = sort_dist[:, k].reshape(-1, 1)
    sig = (Dis_Mat <= np.repeat(threshold, num_samples, axis=1))
    kernel = sig * kernel - np.identity(num_samples)

    return kernel


def get_fused_kernel(omics_list, neighbor_num='default'):
    kernel_list = []
    num_samples = omics_list[0].shape[0]
    fused_kernel = np.zeros((num_samples, num_samples))
    if neighbor_num == 'default':
        neighbor_num = round(num_samples / 10)
        if neighbor_num < 25:
            neighbor_num = 25
        elif neighbor_num > 50:
            neighbor_num = 50
    for i in range(len(omics_list)):
        omics_kernel = knn_kernel(omics_list[i], k=neighbor_num)
        fused_kernel += omics_kernel
        kernel_list.append(omics_kernel)
    return fused_kernel


def get_n_clusters(arr, n_clusters=range(2, 10)):
    # confirm inputs are appropriate
    n_clusters = check_array(n_clusters, ensure_2d=False)
    n_clusters = n_clusters[n_clusters > 1]

    # don't overwrite provided array!
    graph = arr.copy()

    graph = (graph + graph.T) / 2
    graph[np.diag_indices_from(graph)] = 0
    degree = graph.sum(axis=1)
    degree[np.isclose(degree, 0)] += np.spacing(1)
    di = np.diag(1 / np.sqrt(degree))
    laplacian = di @ (np.diag(degree) - graph) @ di

    # perform eigendecomposition and find eigengap
    eigs = np.sort(np.linalg.eig(laplacian)[0])
    eigengap = np.abs(np.diff(eigs))
    eigengap = eigengap * (1 - eigs[:-1]) / (1 - eigs[1:])
    n = eigengap[n_clusters - 1].argsort()[::-1]

    return n_clusters[n[:2]]


def deep_multiomics_clustering(data_arrays,
                               num_features=2000,
                               normalization_type='min-max',
                               neighbor_num='default',
                               epochs_pretrain=1000,
                               epochs_fusion=100,
                               lr=1e-3,
                               weight_xx=1.0,
                               weight_zz=0.05,
                               weight_selfExp=0.2,
                               weight_coef=0.10,
                               omics_weights=None,
                               device='cpu',
                               show_freq=-1,
                               return_coefficient_matrix=False,
                               seed=1234):
    omics_list = []
    for i, data in enumerate(data_arrays):
        # Ensure proper data type conversion
        np_data = np.array(data, dtype=np.float32)
        # Handle any NaN or infinite values
        np_data = np.nan_to_num(np_data, nan=0.0, posinf=0.0, neginf=0.0)
        df = pd.DataFrame(np_data)
        omics_list.append(df)
        # print(f"Data modality {i + 1} shape: {omics_list[-1].shape}, dtype: {omics_list[-1].dtype}")

    # Set random seed for reproducibility
    if seed is not None:
        set_seed(seed)

    num_omics = len(omics_list)
    print(num_omics)

    # Preprocessing
    omics_list = keep_high_var_features(omics_list, num_features=num_features)
    omics_list = normalize_matrix(omics_list, type=normalization_type)

    # Get sample information
    num_samples = omics_list[0].shape[0]
    # samples_id = omics_list[0].index

    # Construct the fused kernel constraint
    fused_kernel = get_fused_kernel(omics_list, neighbor_num)

    # Set omics weights
    if omics_weights is None:
        omics_weights = [1 / num_omics] * num_omics
    w = omics_weights

    # Define input dimensions for each omics
    input_dims = []
    for k in range(num_omics):
        z_dim = min(1024, omics_list[k].shape[1])
        input_dims.append([omics_list[k].shape[1], z_dim, 512])

    # Create temporary directory for model storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Pretrain cycle autoencoders
        pretrained_models = []
        for k in range(num_omics):
            if show_freq > 0:
                print(f"Pretraining autoencoder for omics {k + 1}/{num_omics}")

            cycle_ae = CycleAE(input_dims=input_dims[k])
            cycle_ae.to(device)

            # Train with modified weight_zz for pretraining
            weight_zz_pretrain = 0.5
            train_CycleAE(cycle_ae, omics_list[k], epochs=epochs_pretrain, lr=lr,
                          weight_xx=weight_xx, weight_zz=weight_zz_pretrain,
                          device=device, show_freq=show_freq)

            # Save pretrained model
            model_path = os.path.join(temp_dir, f'omics_{k}_cae.pkl')
            torch.save(cycle_ae.state_dict(), model_path)
            pretrained_models.append(model_path)

        # Create and train DMSCNet
        if show_freq > 0:
            print("Training fusion network...")

        dmscnet = DMSCNet(num_omics, input_dims, num_samples, fused_kernel, w)
        dmscnet.to(device)

        # Load pretrained weights
        for k in range(num_omics):
            cae_state_dict = torch.load(pretrained_models[k])
            dmscnet.CAE[k].load_state_dict(cae_state_dict)

        if show_freq > 0:
            print("Pretrained autoencoder weights loaded successfully.")

        # Train fusion network
        losses, losses_item = train_DMSCNet(dmscnet, omics_list, epochs=epochs_fusion,
                                            num_omics=num_omics, lr=lr, weight_xx=weight_xx,
                                            weight_zz=weight_zz, weight_coef=weight_coef,
                                            weight_selfExp=weight_selfExp,
                                            device=device, show_freq=show_freq)

        # Extract results
        for k in range(num_omics):
            if not isinstance(omics_list[k], torch.Tensor):
                omics_list[k] = torch.tensor(omics_list[k], dtype=torch.float32, device=device)

        Z, Z_selfExp, X_recon, Z_recon = dmscnet.forward(omics_list)
        C = dmscnet.self_expression.Coefficient.detach().to('cpu').numpy()

    # Post-process coefficient matrix
    C = preprocessing.normalize(C, norm='l1', axis=1)
    fused = (C + C.T) / 2

    # Determine optimal number of clusters
    first, second = get_n_clusters(fused)
    cluster_num = max(first, second)

    # Perform spectral clustering
    labels_pred = cluster.spectral_clustering(fused, n_clusters=cluster_num)
    labels_pred = labels_pred.flatten()

    if show_freq > 0:
        print(f"Clustering completed. Found {cluster_num} clusters.")

    if return_coefficient_matrix:
        return labels_pred, fused
    else:
        return labels_pred

###
# mRNA = pd.read_csv("/data/daotran/Cancer_RP/Subtyping/BiB_Subtyping/NewMethods/Deep-Latent-Space-Fusion/aligned_exp.csv",
#                    header=0, index_col=0).T
# miRNA = pd.read_csv("/data/daotran/Cancer_RP/Subtyping/BiB_Subtyping/NewMethods/Deep-Latent-Space-Fusion/aligned_mirna.csv",
#                    header=0, index_col=0).T
#
# omics_list = [mRNA, miRNA]
# cluster = deep_multiomics_clustering(omics_list)



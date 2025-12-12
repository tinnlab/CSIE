import torch.nn as nn
import torch
import numpy as np
from torch.nn import Linear
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans

device = torch.device("cpu")


def set_seed(seed=1234):
    """Set seeds for reproducible results"""
    import random
    # Convert seed to integer to handle R->Python conversion
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_kNNgraph2(data, K_num):
    # each row of data is a sample
    # Ensure data is float32 and handle any NaN values
    data = np.array(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    x_norm = np.reshape(np.sum(np.square(data), 1), [-1, 1])  # column vector
    x_norm2 = np.reshape(np.sum(np.square(data), 1), [1, -1])  # column vector
    dists = x_norm - 2 * np.matmul(data, np.transpose(data)) + x_norm2
    num_sample = data.shape[0]
    graph = np.zeros((num_sample, num_sample), dtype=np.int32)  # Changed to int32
    for i in range(num_sample):
        distance = dists[i, :]
        small_index = np.argsort(distance)
        graph[i, small_index[0:K_num]] = 1
    graph = graph - np.diag(np.diag(graph))
    resultgraph = np.maximum(graph, np.transpose(graph))
    return resultgraph


def comp(g):
    g = np.array(g, dtype=np.float32)  # Ensure float32
    g = g + np.identity(g.shape[0], dtype=np.float32)
    g = torch.tensor(g, dtype=torch.float32)
    d = torch.diag(g.sum(axis=1))
    s = torch.pow(d, -0.5)
    where_are_inf = torch.isinf(s)
    s[where_are_inf] = 0
    s = torch.matmul(torch.matmul(s, g), s)
    return s


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


class Networks(nn.Module):
    def __init__(self, input_data_list, seed=1234):
        super(Networks, self).__init__()
        # Convert seed to integer to handle R->Python conversion
        seed = int(seed)
        torch.manual_seed(seed)

        self.n_modalities = len(input_data_list)

        # Store dimensions for each modality
        self.modality_dims = []

        # Create encoders and decoders for each modality
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.weights = nn.ParameterList()  # For adjacency prediction
        self.fusion_weights = nn.ParameterList()  # For reconstruction

        # Determine shared representation dimension based on the smallest modality
        # or use a fixed dimension
        min_features = min(data.shape[1] for data in input_data_list)
        shared_dim = int(min_features * 0.8)

        for i, input_data in enumerate(input_data_list):
            input_dim = input_data.shape[1]

            # Create dimension structure for this modality
            hidden_dim = int(input_dim * 0.8)
            dims = [hidden_dim, shared_dim]
            self.modality_dims.append(dims)

            # Create encoder layers
            encoder_layers = nn.Sequential(
                Linear(input_dim, dims[0]),
                nn.Tanh(),
                Linear(dims[0], dims[1]),
                nn.Tanh()
            )
            self.encoders.append(encoder_layers)

            # Create decoder layers
            decoder_layers = nn.Sequential(
                Linear(dims[1], dims[0]),
                nn.Tanh(),
                Linear(dims[0], input_dim),
                nn.Tanh()
            )
            self.decoders.append(decoder_layers)

            # Create weight matrices for adjacency prediction and fusion
            self.weights.append(
                torch.nn.init.xavier_uniform_(nn.Parameter(torch.FloatTensor(dims[1], dims[1])))
            )
            self.fusion_weights.append(
                torch.nn.init.xavier_uniform_(nn.Parameter(torch.FloatTensor(dims[1], dims[1])))
            )

        # Set seed again after parameter initialization
        torch.manual_seed(seed)

    def forward(self, input_list, we, normalized_graphs, kmeans_seed=1234):
        # Convert seed to integer
        kmeans_seed = int(kmeans_seed)
        # Encode all modalities
        encoded_representations = []

        for i, (input_data, norm_graph) in enumerate(zip(input_list, normalized_graphs)):
            # Apply graph convolution during encoding
            h1 = torch.tanh(self.encoders[i][0](torch.matmul(norm_graph, input_data)))
            h2 = torch.tanh(self.encoders[i][2](torch.matmul(norm_graph, h1)))
            encoded_representations.append(h2)

        # Create shared representation by weighted combination
        weighted_sum = sum(torch.diag(we[:, i]).mm(encoded_representations[i])
                           for i in range(self.n_modalities))

        weights_sum = 1 / torch.sum(we, 1)
        z = torch.diag(weights_sum).mm(weighted_sum)

        # Predict adjacency matrices
        predicted_adjacencies = []
        for i in range(self.n_modalities):
            adj_pred = torch.sigmoid(torch.matmul(torch.matmul(z, self.weights[i]), z.T))
            predicted_adjacencies.append(adj_pred)

        # Decode to reconstruct original data
        reconstructed_data = []
        for i, norm_graph in enumerate(normalized_graphs):
            # Apply fusion weight and then decode with graph convolution
            h_fused = torch.tanh(torch.matmul(z, self.fusion_weights[i]))
            h_dec1 = torch.tanh(self.decoders[i][0](torch.matmul(norm_graph, h_fused)))
            h_dec2 = torch.tanh(self.decoders[i][2](torch.matmul(norm_graph, h_dec1)))
            reconstructed_data.append(h_dec2)

        # Clustering using spectral clustering with eigengap method
        z_np = z.detach().numpy()

        # Create affinity matrix using RBF kernel
        affinity_matrix = rbf_kernel(z_np)

        # Compute eigenvalues for eigengap method
        eigenvalues, _ = np.linalg.eigh(affinity_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order

        # Modified eigengap method to determine number of clusters
        max_clusters = 10  # Limit search space
        eigengap_scores = []

        for i in range(1, max_clusters):
            if i < len(eigenvalues):
                eigengap = (eigenvalues[i - 1] - eigenvalues[i]) * i
                eigengap_scores.append(eigengap)
            else:
                eigengap_scores.append(0)

        n_clusters = np.argmax(eigengap_scores) + 1
        n_clusters = max(2, min(n_clusters, max_clusters))  # Ensure reasonable range

        # Perform spectral clustering
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                                      random_state=kmeans_seed)
        y_pred_ = spectral.fit_predict(affinity_matrix)

        # Compute cluster centers by taking mean of points in each cluster
        cluster_centers = []
        for cluster_id in range(n_clusters):
            cluster_mask = (y_pred_ == cluster_id)
            if np.sum(cluster_mask) > 0:
                cluster_center = np.mean(z_np[cluster_mask], axis=0)
                cluster_centers.append(cluster_center)
            else:
                # Handle empty cluster case
                cluster_centers.append(np.mean(z_np, axis=0))

        cluster_layer = torch.tensor(np.array(cluster_centers), dtype=torch.float32).to(device)

        # Compute soft cluster assignments
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - cluster_layer, 2), 2))
        q = q.pow(1)
        q = (q.t() / torch.sum(q, 1)).t()
        p = target_distribution(q)

        return reconstructed_data, predicted_adjacencies, z, p, q, cluster_layer
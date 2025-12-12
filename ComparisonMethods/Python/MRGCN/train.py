# import sys

# sys.path.append("/data/daotran/Cancer_RP/Subtyping/BiB_Subtyping/NewMethods/MRGCN_New")

from torch.nn import functional as F
from network import *
from dataloader import create_data_loader_from_arrays


def run_clustering_with_data(*data_arrays, seed=1234):
    """
    Run clustering with variable number of data modalities passed from R

    Args:
        *data_arrays: Variable number of data arrays (each representing a different datatype)
        seed: Random seed for reproducibility
    """

    # Convert seed to integer to handle R->Python conversion
    seed = int(seed)
    set_seed(seed)

    # Convert R data to numpy arrays with explicit dtype handling
    processed_data = []
    for i, data in enumerate(data_arrays):
        # Ensure proper data type conversion
        np_data = np.array(data, dtype=np.float32)
        # Handle any NaN or infinite values
        np_data = np.nan_to_num(np_data, nan=0.0, posinf=0.0, neginf=0.0)
        processed_data.append(np_data)
        print(f"Data modality {i + 1} shape: {processed_data[-1].shape}, dtype: {processed_data[-1].dtype}")

    n_modalities = len(processed_data)
    print(f"Number of data modalities: {n_modalities}")

    # Create data loader from arrays
    data_loader_train = create_data_loader_from_arrays(
        *processed_data,
        batch_size=processed_data[0].shape[0],
        shuffle=False
    )

    # Get the processed data for graph creation
    dataset = data_loader_train.dataset
    processed_imgs = dataset.processed_data  # This will be a list of processed arrays

    # Create weight matrix for all modalities
    indicators = []
    for img in processed_imgs:
        # Ensure boolean operations work correctly
        img_array = np.array(img, dtype=np.float32)
        ind = np.any(img_array != 0, axis=1).astype(np.float32)  # Changed to float32
        indicators.append(ind)

    we = np.array(list(zip(*indicators)), dtype=np.float32)
    we = torch.tensor(we, dtype=torch.float32)

    # Create graphs for all modalities
    graphs = []
    normalized_graphs = []
    for img in processed_imgs:
        g = get_kNNgraph2(img, 10)
        s = comp(g)
        graphs.append(g)
        normalized_graphs.append(s)

    # Initialize model
    device = torch.device("cpu")
    model = Networks(processed_imgs, seed=seed).to(device)
    criterion = torch.nn.MSELoss(reduction='sum')

    # Training parameters
    regg2 = [1]  # adjacency loss weight
    regg3 = [1]  # clustering loss weight
    n_epochs2 = 100

    cluster_assignments = None

    # Training loop
    for r in range(len(regg2)):
        reg2 = regg2[r]
        for h in range(len(regg3)):
            reg3 = regg3[h]
            set_seed(seed)
            model = Networks(processed_imgs, seed=seed).to(device)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=0.001, weight_decay=0.0)

            for epoch in range(n_epochs2):
                for data in data_loader_train:
                    # data now contains variable number of modalities
                    inputs = [d.clone().detach() for d in data]

                    # Forward pass
                    reconstructed_data, predicted_adjacencies, z, p, q, cluster_layer = model.forward(
                        inputs, we, normalized_graphs, kmeans_seed=seed
                    )

                    # Get hard cluster assignments from soft assignments
                    cluster_assignments = torch.argmax(q, dim=1).cpu().detach().numpy()

                    # Calculate losses
                    # Reconstruction loss for all modalities
                    loss_x = sum(criterion(recon, inp) for recon, inp in zip(reconstructed_data, inputs))

                    # Adjacency prediction loss for all modalities
                    loss_a = sum(criterion(pred_adj, torch.Tensor(graph))
                                 for pred_adj, graph in zip(predicted_adjacencies, graphs))

                    # KL divergence loss for clustering
                    loss_kl = F.kl_div(q.log(), p, reduction='batchmean')

                    # Total loss
                    loss = loss_x + reg2 * loss_a + reg3 * loss_kl

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    # Return results
    return {
        'cluster_assignments': cluster_assignments,
        'soft_assignments': q.cpu().detach().numpy(),
        'shared_representation': z.cpu().detach().numpy(),
        'cluster_centers': cluster_layer.cpu().detach().numpy(),
        'n_clusters': cluster_layer.shape[0],
        'n_modalities': n_modalities
    }
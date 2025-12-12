import torch.nn as nn
import torch


class FlexibleMultiOmicsNetwork(nn.Module):
    def __init__(self, feature_dims, n_samples, latent_dim=15, kernel_size=5):
        """
        Initialize flexible multi-omics autoencoder

        Parameters:
        -----------
        feature_dims : list of int
            Number of features for each modality
        n_samples : int
            Number of samples in the dataset
        latent_dim : int, optional
            Latent dimension for each encoder (default: 15)
        kernel_size : int, optional
            Kernel size for convolution (default: 5)
        """
        super(FlexibleMultiOmicsNetwork, self).__init__()

        self.n_modalities = len(feature_dims)
        self.feature_dims = feature_dims
        self.n_samples = n_samples
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size

        # Create encoders and decoders for each modality
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i, feat_dim in enumerate(feature_dims):
            # Encoder
            encoder = nn.Sequential(
                nn.Conv2d(1, latent_dim, kernel_size=(kernel_size, 1),
                          stride=kernel_size, padding=0, bias=True),
                nn.ReLU(),
            )

            # Decoder
            decoder = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, 1, kernel_size=(kernel_size, 1),
                                   stride=kernel_size, padding=0, bias=False),
                nn.ReLU(),
            )

            self.encoders.append(encoder)
            self.decoders.append(decoder)

        # Learnable sample relationship matrix
        self.weight = nn.Parameter(1.0e-4 * torch.ones(n_samples, n_samples))

    def forward(self, inputs):
        """
        Standard autoencoder forward pass

        Parameters:
        -----------
        inputs : list of tensors
            Each tensor should be (batch_size, 1, features, 1)

        Returns:
        --------
        outputs : list of tensors
            Reconstructed inputs
        """
        outputs = []

        for i, (input_tensor, encoder, decoder) in enumerate(zip(inputs, self.encoders, self.decoders)):
            encoded = encoder(input_tensor)
            decoded = decoder(encoded)
            outputs.append(decoded)

        return outputs

    def forward2(self, inputs):
        """
        Relationship-aware forward pass

        Parameters:
        -----------
        inputs : list of tensors
            Each tensor should be (batch_size, 1, features, 1)

        Returns:
        --------
        z_list : list of tensors
            Original latent representations
        outputs : list of tensors
            Reconstructed inputs
        zcoef_list : list of tensors
            Relationship-modified latent representations
        coef : tensor
            Sample relationship matrix (without diagonal)
        """
        # Remove diagonal from weight matrix
        coef = self.weight - torch.diag(torch.diag(self.weight))

        z_list = []
        zcoef_list = []
        outputs = []

        for i, (input_tensor, encoder, decoder) in enumerate(zip(inputs, self.encoders, self.decoders)):
            # Encode
            z = encoder(input_tensor)
            z_flat = z.view(self.n_samples, -1)
            z_list.append(z_flat)

            # Apply sample relationships
            zcoef = torch.matmul(coef, z_flat)
            zcoef_list.append(zcoef)

            # Reshape and decode
            zcoef_reshaped = zcoef.view(self.n_samples, self.latent_dim, -1, 1)
            output = decoder(zcoef_reshaped)
            outputs.append(output)

        return z_list, outputs, zcoef_list, coef

    def get_feature_shapes(self):
        """Helper method to get expected feature shapes after encoding"""
        shapes = []
        for feat_dim in self.feature_dims:
            encoded_dim = feat_dim // self.kernel_size
            shapes.append(encoded_dim)
        return shapes
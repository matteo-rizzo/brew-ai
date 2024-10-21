import torch
from torch import nn

from src.classes.evaluation.periodicity.embedding.FeatureEmbedding import FeatureEmbedding
from src.classes.evaluation.periodicity.models.autopnp.AutoPNPLayer import AutoPNPLayer
from src.classes.evaluation.periodicity.models.mlp.MLPRegressor import MLPRegressor


class AutoPNPNet(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_fourier_features: int,
            num_chebyshev_terms: int,
            use_feature_embeddings: bool = True
    ):
        """
        AutoPNPNet that integrates Fourier and Chebyshev layers with feature selection, embeddings, and an MLP regressor.

        :param input_size: Size of the input features.
        :param num_fourier_features: Number of Fourier features to generate.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        :param use_feature_embeddings: Whether to use a feature embedding layer.
        """
        super(AutoPNPNet, self).__init__()

        # Configurations
        self.use_feature_embeddings = use_feature_embeddings

        # Feature embedding (optional)
        if self.use_feature_embeddings:
            self.feature_embedding = FeatureEmbedding(input_size)
            input_size = self.feature_embedding.get_embedding_dim()

        # Fourier and Chebyshev layers
        self.autopnp_layer = AutoPNPLayer(input_size, num_fourier_features, num_chebyshev_terms)

        # Feature dimensions after transformations
        total_fourier_features = num_fourier_features * 2 * input_size  # Fourier has sine and cosine
        total_chebyshev_features = num_chebyshev_terms * input_size
        total_features = total_fourier_features + total_chebyshev_features

        self.mlp_regressor = MLPRegressor(input_size=total_features)

        # Residual connection
        self.residual_layer = nn.Linear(total_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Apply feature embeddings (if enabled)
        if self.use_feature_embeddings:
            x = self.feature_embedding(x)

        # Combine Fourier and Chebyshev transformed features
        x_combined = self.autopnp_layer(x)

        # Apply residual connection and MLP regressor
        residual = self.residual_layer(x_combined)
        out = self.mlp_regressor(x_combined)
        out += residual

        return out

import torch
from torch import nn

from src.classes.evaluation.periodicity.factories.FeatureEmbeddingFactory import FeatureEmbeddingFactory
from src.classes.evaluation.periodicity.models.autopnp.AutoPNPLayer import AutoPNPLayer
from src.classes.evaluation.periodicity.models.categorical.CategoricalMLP import CategoricalMLP
from src.classes.evaluation.periodicity.models.regressor.TabMLPRegressor import TabMLPRegressor


class TabAutoPNPNet(nn.Module):
    def __init__(
            self,
            continuous_input_size: int,
            categorical_input_size: int,
            num_fourier_features: int,
            num_chebyshev_terms: int,
            hidden_size: int,
            use_feature_embeddings: bool = False,
            embedding_type: str = "linear"
    ):
        """
        TabAutoPNPNet: Extends AutoPNPNet to support both continuous and one-hot encoded categorical features.

        :param continuous_input_size: Number of continuous input features.
        :param categorical_input_size: Total number of one-hot encoded categorical features.
        :param num_fourier_features: Number of Fourier features per continuous input feature.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        :param hidden_size: Size of hidden layers.
        """
        super(TabAutoPNPNet, self).__init__()

        continuous_input_size = continuous_input_size
        categorical_input_size = categorical_input_size

        # Configurations
        self.use_feature_embeddings = use_feature_embeddings

        # Feature embedding (optional)
        if self.use_feature_embeddings:
            self.feature_embedding = FeatureEmbeddingFactory.get_feature_embedding(embedding_type,
                                                                                   continuous_input_size)
            continuous_input_size = self.feature_embedding.get_embedding_dim()

        # Fourier and Chebyshev layers
        self.autopnp_layer = AutoPNPLayer(continuous_input_size, num_fourier_features, num_chebyshev_terms)

        # MLP for one-hot encoded categorical features
        self.categorical_layer = CategoricalMLP(categorical_input_size, hidden_size)

        # Feature dimensions after transformations
        total_fourier_features = num_fourier_features * 2 * continuous_input_size  # Fourier has sine and cosine
        total_chebyshev_features = num_chebyshev_terms * continuous_input_size
        categorical_input_size = hidden_size
        total_features = total_fourier_features + total_chebyshev_features + categorical_input_size

        # TabMLPRegressor for combined processing
        self.regressor = TabMLPRegressor(total_features)

        # Residual connection layer for continuous features
        self.residual_layer = nn.Linear(total_fourier_features + total_chebyshev_features, 1)

    def forward(self, x_continuous, x_categorical):

        # Apply feature embeddings (if enabled)
        if self.use_feature_embeddings:
            x_continuous = self.feature_embedding(x_continuous)

        # Combine Fourier and Chebyshev transformed features
        x_continuous_combined = self.autopnp_layer(x_continuous)

        # Residual connection for continuous features
        residual = self.residual_layer(x_continuous_combined)

        # Process one-hot encoded categorical features through MLP
        x_categorical_processed = self.categorical_layer(x_categorical)  # Shape: [batch_size, hidden_size]

        # Combine continuous and categorical features
        x_combined = torch.cat([x_continuous_combined, x_categorical_processed], dim=1)

        # Pass through fully connected layers
        out = self.regressor(x_combined)

        # Add residual (only from continuous features)
        out += residual

        return out

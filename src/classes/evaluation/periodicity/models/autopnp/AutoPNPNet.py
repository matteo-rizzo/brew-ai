import torch
from torch import nn

from src.classes.evaluation.periodicity.factories.FeatureEmbeddingFactory import FeatureEmbeddingFactory
from src.classes.evaluation.periodicity.models.autopnp.AutoPNPLayer import AutoPNPLayer
from src.classes.evaluation.periodicity.models.regressor.MLPRegressor import MLPRegressor


class AutoPNPBlock(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_fourier_features: int,
            num_chebyshev_terms: int,
            num_layers: int = 1,
            compression_dim: int = None,
            feature_selector: str = "default",
            feature_selection_before_transform: bool = False,
            dropout_prob: float = 0.1
    ):
        """
        AutoPNPBlock that stacks multiple AutoPNPLayer layers with optional compression layers,
        batch normalization, and dropout.

        :param input_size: Size of the input features.
        :param num_fourier_features: Number of Fourier features to generate in each layer.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms in each layer.
        :param num_layers: Number of AutoPNPLayer layers to stack.
        :param compression_dim: Dimension to compress features to between layers.
        :param feature_selector: Type of feature selector to use.
        :param feature_selection_before_transform: Whether to apply feature selection before transformations.
        :param dropout_prob: Probability of dropout after batch normalization (0 to 1).
        """
        super(AutoPNPBlock, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.compress_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.dropout_prob = dropout_prob

        self.input_size = input_size
        current_input_size = input_size

        for i in range(num_layers):
            # Create an AutoPNPLayer
            autopnp_layer = AutoPNPLayer(
                input_size=current_input_size,
                num_fourier_features=num_fourier_features,
                num_chebyshev_terms=num_chebyshev_terms,
                feature_selector=feature_selector,
                feature_selection_before_transform=feature_selection_before_transform
            )
            self.layers.append(autopnp_layer)

            # Calculate output feature dimension from AutoPNPLayer
            total_fourier_features = num_fourier_features * 2 * current_input_size  # Fourier has sine and cosine
            total_chebyshev_features = num_chebyshev_terms * current_input_size
            total_features = total_fourier_features + total_chebyshev_features

            # Add compression layer to reduce back to input_size (except after the last layer)
            if i < num_layers - 1:
                compress_dim = compression_dim if compression_dim is not None else current_input_size
                compress_layer = nn.Linear(total_features, compress_dim)
                self.compress_layers.append(compress_layer)

                bn_layer = nn.BatchNorm1d(compress_dim)
                self.bn_layers.append(bn_layer)

                dropout_layer = nn.Dropout(p=self.dropout_prob)
                self.dropout_layers.append(dropout_layer)

                current_input_size = compress_dim
            else:
                # For the last layer, store the total_features as output_dim
                self.output_dim = total_features

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.compress_layers[i](x)
                x = self.bn_layers[i](x)
                x = self.dropout_layers[i](x)
        return x


class AutoPNPNet(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_fourier_features: int,
            num_chebyshev_terms: int,
            num_layers: int = 1,
            compression_dim: int = 128,
            use_feature_embeddings: bool = False,
            embedding_type="deep",
            feature_selector: str = "default",
            feature_selection_before_transform: bool = False,
    ):
        """
        AutoPNPNet that integrates the AutoPNPBlock with feature selection, embeddings, and an MLP regressor.

        :param input_size: Size of the input features.
        :param num_fourier_features: Number of Fourier features to generate.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        :param num_layers: Number of AutoPNPLayer layers to stack.
        :param compression_dim: Dimension to compress features to between layers.
        :param use_feature_embeddings: Whether to use a feature embedding layer.
        :param embedding_type: Type of embedding to use if feature embeddings are enabled.
        :param feature_selector: Type of feature selector to use.
        :param feature_selection_before_transform: Whether to apply feature selection before transformations.
        """
        super(AutoPNPNet, self).__init__()

        # Configurations
        self.use_feature_embeddings = use_feature_embeddings

        # Feature embedding (optional)
        if self.use_feature_embeddings:
            self.feature_embedding = FeatureEmbeddingFactory.get_feature_embedding(embedding_type, input_size)
            input_size = self.feature_embedding.get_embedding_dim()

        # Create the AutoPNPBlock
        self.autopnp_block = AutoPNPBlock(
            input_size=input_size,
            num_fourier_features=num_fourier_features,
            num_chebyshev_terms=num_chebyshev_terms,
            num_layers=num_layers,
            compression_dim=compression_dim,
            feature_selector=feature_selector,
            feature_selection_before_transform=feature_selection_before_transform
        )

        # Feature dimensions after transformations
        total_features = self.autopnp_block.output_dim

        self.mlp_regressor = MLPRegressor(input_size=total_features)

        # Residual connection
        #self.residual_layer = nn.Linear(total_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Apply feature embeddings (if enabled)
        if self.use_feature_embeddings:
            x = self.feature_embedding(x)

        # Pass through the AutoPNPBlock
        x_combined = self.autopnp_block(x)

        # Apply residual connection and MLP regressor
        #residual = self.residual_layer(x_combined)
        out = self.mlp_regressor(x_combined)
        #out += residual

        return out

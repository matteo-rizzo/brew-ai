from torch import Tensor
from torch import nn

from src.classes.evaluation.periodicity.models.chebyshev.AdaptiveChebyshevConvLayer import AdaptiveChebyshevConvLayer
from src.classes.evaluation.periodicity.models.chebyshev.AdaptiveChebyshevLayer import AdaptiveChebyshevLayer
from src.classes.evaluation.periodicity.models.chebyshev.MultiHeahAdaptiveChebyshevLayer import \
    MultiHeadAdaptiveChebyshevLayer
from src.classes.evaluation.periodicity.models.regressor.MLPRegressor import MLPRegressor


class ChebyshevBlock(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_chebyshev_terms: int,
            num_layers: int = 1,
            compression_dim: int = None,
            use_batch_norm: bool = False,
            use_layer_norm: bool = True,
            dropout_prob: float = 0.5
    ):
        """
        ChebyshevBlock that stacks multiple AdaptiveChebyshevConvLayer layers with optional compression layers,
        batch normalization, layer normalization, and dropout.

        :param input_size: Size of the input features.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms in each layer.
        :param num_layers: Number of AdaptiveChebyshevConvLayer layers to stack.
        :param compression_dim: Dimension to compress features to between layers.
        :param use_batch_norm: Whether to use batch normalization after compression layers.
        :param use_layer_norm: Whether to use layer normalization after compression layers.
        :param dropout_prob: Probability of dropout after normalization layers (0 to 1).
        """
        super(ChebyshevBlock, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.compress_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.dropout_prob = dropout_prob

        self.input_size = input_size
        current_input_size = input_size

        for i in range(num_layers):
            # Create an AdaptiveChebyshevConvLayer
            chebyshev_layer = AdaptiveChebyshevLayer(current_input_size, num_chebyshev_terms)
            self.layers.append(chebyshev_layer)

            # Calculate output feature dimension from Chebyshev layer
            total_features = num_chebyshev_terms * current_input_size

            # Add compression layer to reduce back to input_size (except after the last layer)
            if i < num_layers - 1:
                compress_dim = compression_dim if compression_dim is not None else current_input_size
                compress_layer = nn.Linear(total_features, compress_dim)
                self.compress_layers.append(compress_layer)

                # Add normalization layer
                if self.use_batch_norm:
                    norm_layer = nn.BatchNorm1d(compress_dim)
                elif self.use_layer_norm:
                    norm_layer = nn.LayerNorm(compress_dim)
                else:
                    norm_layer = None
                self.norm_layers.append(norm_layer)

                # Add dropout layer
                if self.dropout_prob > 0:
                    dropout_layer = nn.Dropout(p=self.dropout_prob)
                else:
                    dropout_layer = None
                self.dropout_layers.append(dropout_layer)

                current_input_size = compress_dim
            else:
                # For the last layer, store the total_features as output_dim
                self.output_dim = total_features

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)  # Apply Chebyshev transformation

            if i < self.num_layers - 1:
                x = self.compress_layers[i](x)  # Compress features

                # Apply normalization
                if self.norm_layers[i] is not None:
                    x = self.norm_layers[i](x)

                # Apply dropout
                if self.dropout_layers[i] is not None:
                    x = self.dropout_layers[i](x)
        return x


class ChebyshevNet(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_chebyshev_terms: int,
            num_layers: int = 1,
            compression_dim: int = 128,
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
            dropout_prob: float = 0.2
    ):
        """
        ChebyshevNet integrates Chebyshev transformations with an MLP regressor.

        :param input_size: Size of the input features.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        :param num_layers: Number of Chebyshev layers to stack.
        :param compression_dim: Dimension to compress features to between layers.
        :param use_batch_norm: Whether to use batch normalization after compression layers.
        :param use_layer_norm: Whether to use layer normalization after compression layers.
        :param dropout_prob: Probability of dropout after normalization layers (0 to 1).
        """
        super(ChebyshevNet, self).__init__()

        # Chebyshev Block
        self.chebyshev_block = ChebyshevBlock(
            input_size=input_size,
            num_chebyshev_terms=num_chebyshev_terms,
            num_layers=num_layers,
            compression_dim=compression_dim,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_prob=dropout_prob
        )

        total_features = self.chebyshev_block.output_dim  # Total number of Chebyshev features after the block

        # MLPRegressor for handling the fully connected layers
        self.regressor = MLPRegressor(total_features)

    def forward(self, x: Tensor) -> Tensor:
        # Apply Chebyshev transformation through the ChebyshevBlock
        x_chebyshev = self.chebyshev_block(x)

        # Pass through the MLP regressor
        out = self.regressor(x_chebyshev)

        return out

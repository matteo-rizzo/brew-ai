import torch
from torch import nn

from src.classes.evaluation.periodicity.models.fourier.FourierLayer import FourierLayer
from src.classes.evaluation.periodicity.models.regressor.MLPRegressor import MLPRegressor


class FourierBlock(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_fourier_features: int,
            num_layers: int = 1,
            compression_dim: int = None,
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
            dropout_prob: float = 0.2
    ):
        """
        FourierBlock that stacks multiple FourierLayer layers with optional compression layers,
        batch normalization, layer normalization, and dropout.

        :param input_size: Size of the input features.
        :param num_fourier_features: Number of Fourier features to generate in each layer.
        :param num_layers: Number of FourierLayer layers to stack.
        :param compression_dim: Dimension to compress features to between layers.
        :param use_batch_norm: Whether to use batch normalization after compression layers.
        :param use_layer_norm: Whether to use layer normalization after compression layers.
        :param dropout_prob: Probability of dropout after normalization layers (0 to 1).
        """
        super(FourierBlock, self).__init__()

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
            # Create a FourierLayer
            fourier_layer = FourierLayer(current_input_size, num_fourier_features)
            self.layers.append(fourier_layer)

            # Calculate output feature dimension from FourierLayer
            total_features = num_fourier_features * 2 * current_input_size  # sine and cosine outputs

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
            x = layer(x)  # Apply Fourier transformation

            if i < self.num_layers - 1:
                x = self.compress_layers[i](x)  # Compress features

                # Apply normalization
                if self.norm_layers[i] is not None:
                    x = self.norm_layers[i](x)

                # Apply dropout
                if self.dropout_layers[i] is not None:
                    x = self.dropout_layers[i](x)
        return x


class FourierNet(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_fourier_features: int,
            num_layers: int = 1,
            compression_dim: int = 128,
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
            dropout_prob: float = 0.2
    ):
        """
        FourierNet integrates Fourier transformations with an MLP regressor.

        :param input_size: Size of the input features.
        :param num_fourier_features: Number of Fourier features to learn.
        :param num_layers: Number of FourierLayer layers to stack.
        :param compression_dim: Dimension to compress features to between layers.
        :param use_batch_norm: Whether to use batch normalization after compression layers.
        :param use_layer_norm: Whether to use layer normalization after compression layers.
        :param dropout_prob: Probability of dropout after normalization layers (0 to 1).
        """
        super(FourierNet, self).__init__()

        # Fourier Block
        self.fourier_block = FourierBlock(
            input_size=input_size,
            num_fourier_features=num_fourier_features,
            num_layers=num_layers,
            compression_dim=compression_dim,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_prob=dropout_prob
        )

        total_features = self.fourier_block.output_dim

        # MLPRegressor for handling the fully connected layers
        self.regressor = MLPRegressor(total_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Fourier transformation through the FourierBlock
        x_fourier = self.fourier_block(x)

        # Pass through the MLP regressor
        out = self.regressor(x_fourier)

        return out

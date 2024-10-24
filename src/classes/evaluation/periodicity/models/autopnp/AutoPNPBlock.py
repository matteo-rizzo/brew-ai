from torch import nn

from src.classes.evaluation.periodicity.models.autopnp.AutoPNPEncoder import AutoPNPEncoder


class AutoPNPBlock(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_fourier_features: int,
            num_chebyshev_terms: int,
            num_layers: int = 1,
            compression_dim: int = None,
            dropout_prob: float = 0.1
    ):
        """
        AutoPNPBlock that stacks multiple AutoPNPEncoder layers with optional compression layers,
        batch normalization, and dropout.

        :param input_size: Size of the input features.
        :param num_fourier_features: Number of Fourier features to generate in each layer.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms in each layer.
        :param num_layers: Number of AutoPNPEncoder layers to stack.
        :param compression_dim: Dimension to compress features to between layers.
        :param dropout_prob: Probability of dropout after batch normalization (0 to 1).
        """
        super(AutoPNPBlock, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.compress_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.dropout_prob = dropout_prob

        current_input_size = input_size

        for i in range(num_layers):
            # Create an AutoPNPEncoder
            self.layers.append(AutoPNPEncoder(current_input_size, num_fourier_features, num_chebyshev_terms))

            # Calculate output feature dimension from AutoPNPEncoder
            total_fourier_features = num_fourier_features * 2 * current_input_size  # Fourier has sine and cosine
            total_chebyshev_features = num_chebyshev_terms * current_input_size
            total_features = total_fourier_features + total_chebyshev_features

            # Add compression layer to reduce back to input_size (except after the last layer)
            if i < num_layers - 1:
                compress_dim = compression_dim if compression_dim is not None else current_input_size
                self.compress_layers.append(nn.Linear(total_features, compress_dim))
                self.bn_layers.append(nn.BatchNorm1d(compress_dim))
                self.dropout_layers.append(nn.Dropout(p=self.dropout_prob))
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

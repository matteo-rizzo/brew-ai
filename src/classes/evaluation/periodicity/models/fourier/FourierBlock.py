from torch import nn

from src.classes.evaluation.periodicity.models.fourier.FourierEncoder import FourierEncoder


class FourierBlock(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_fourier_features: int,
            num_layers: int = 1,
            compression_dim: int = None,
            dropout_prob: float = 0.2
    ):
        """
        FourierBlock that stacks multiple FourierEncoder layers with optional compression layers,
        batch normalization, layer normalization, and dropout.

        :param input_size: Size of the input features.
        :param num_fourier_features: Number of Fourier features to generate in each layer.
        :param num_layers: Number of FourierEncoder layers to stack.
        :param compression_dim: Dimension to compress features to between layers.
        :param dropout_prob: Probability of dropout after normalization layers (0 to 1).
        """
        super(FourierBlock, self).__init__()

        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.layers = nn.ModuleList()
        self.compress_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        current_input_size = input_size

        for i in range(num_layers):
            # Create a FourierEncoder
            self.layers.append(FourierEncoder(current_input_size, num_fourier_features))

            # Calculate output feature dimension from FourierEncoder
            total_features = num_fourier_features * 2 * current_input_size # sine and cosine outputs

            # Add compression layer to reduce back to input_size (except after the last layer)
            if i < num_layers - 1:
                compress_dim = compression_dim if compression_dim is not None else current_input_size
                self.compress_layers.append(nn.Linear(total_features, compress_dim))
                self.norm_layers.append(nn.BatchNorm1d(compress_dim))
                self.dropout_layers.append(nn.Dropout(p=self.dropout_prob))
                current_input_size = compress_dim
            else:
                # For the last layer, store the total_features as output_dim
                self.output_dim = total_features

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)  # Apply Fourier transformation
            if i < self.num_layers - 1:
                x = self.compress_layers[i](x)  # Compress features
                x = self.norm_layers[i](x)
                x = self.dropout_layers[i](x)
        return x

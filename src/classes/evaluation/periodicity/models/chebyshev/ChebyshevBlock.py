import torch
from torch import nn

from src.classes.evaluation.periodicity.models.chebyshev.ChebyshevEncoder import ChebyshevEncoder


class ChebyshevBlock(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_chebyshev_terms: int,
            num_layers: int = 1,
            compression_dim: int = None,
            dropout_prob: float = 0.5
    ):
        """
        ChebyshevBlock that stacks multiple ChebyshevEncoder layers with optional compression, batch normalization,
        and dropout.

        :param input_size: Size of the input features.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms in each layer.
        :param num_layers: Number of ChebyshevEncoder layers to stack.
        :param compression_dim: Dimension to compress features between layers.
        :param dropout_prob: Probability of dropout after normalization layers.
        """
        super(ChebyshevBlock, self).__init__()

        self.layers = nn.ModuleList()
        self.compress_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        self.num_layers = num_layers
        current_input_size = input_size

        for i in range(num_layers):
            # Create the ChebyshevEncoder layer
            self.layers.append(ChebyshevEncoder(current_input_size, num_chebyshev_terms))

            # Compute output feature dimension from the Chebyshev layer
            total_features = num_chebyshev_terms * current_input_size

            if i < num_layers - 1:
                # Add compression layer (reduce back to compression_dim or input_size)
                compress_dim = compression_dim if compression_dim is not None else current_input_size
                self.compress_layers.append(nn.Linear(total_features, compress_dim))
                self.norm_layers.append(nn.BatchNorm1d(compress_dim))
                self.dropout_layers.append(nn.Dropout(p=dropout_prob))
                current_input_size = compress_dim
            else:
                # Store the final output dimension
                self.output_dim = total_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ChebyshevBlock.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Output tensor after stacked Chebyshev transformations, compression, and optional normalization/dropout.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)  # Apply the Chebyshev transformation
            if i < self.num_layers - 1:
                x = self.compress_layers[i](x)
                x = self.norm_layers[i](x)
                x = self.dropout_layers[i](x)
        return x

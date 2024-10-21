import torch
from torch import nn


class TransformerRegressor(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int = 8,
            num_heads: int = 4,
            num_layers: int = 2,
            num_outputs: int = 1,
            dropout_prob: float = 0.2,
    ):
        """
        Transformer-based regression model for 2D input (tabular data).

        :param input_size: The size of the input features.
        :param hidden_size: The size of hidden layers in the transformer.
        :param num_heads: Number of attention heads in the Transformer layers.
        :param num_layers: Number of Transformer layers.
        :param num_outputs: Number of outputs (regression targets).
        :param dropout_prob: Dropout probability for regularization.
        """
        super(TransformerRegressor, self).__init__()

        # Linear projection of input data to hidden size
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Transformer layers with batch_first=True (to handle [batch_size, input_size] format)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout_prob,
                batch_first=True  # Keep batch dimension first
            ),
            num_layers=num_layers
        )

        # Final linear layer for regression output
        self.output_layer = nn.Linear(hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer Regressor for 2D input.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Output tensor of shape [batch_size, num_outputs].
        """
        # Project input to hidden size
        x = self.input_projection(x)  # Shape: [batch_size, hidden_size]

        # Transformer expects a 3D input [batch_size, 1, hidden_size] for non-sequential data
        x = x.unsqueeze(1)  # Add a "sequence" dimension: [batch_size, 1, hidden_size]

        # Apply Transformer Encoder
        x = self.transformer_encoder(x)  # Shape: [batch_size, 1, hidden_size]

        # Remove the "sequence" dimension
        x = x.squeeze(1)  # Shape: [batch_size, hidden_size]

        # Final output layer for regression
        out = self.output_layer(x)  # Shape: [batch_size, num_outputs]
        return out

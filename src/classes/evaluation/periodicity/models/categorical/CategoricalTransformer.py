from typing import Optional

import torch
from torch import nn


class CategoricalTransformer(nn.Module):
    def __init__(
            self,
            input_size: int,  # Size of the one-hot encoded input (number of categories/features)
            embedding_dim: int = 16,
            hidden_size: int = 64,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout_prob: float = 0.2,
    ):
        """
        CategoricalTransformer for processing one-hot encoded categorical data using a transformer-based architecture.

        :param input_size: The size of the input features (number of one-hot encoded features).
        :param embedding_dim: Dimension of the hidden representation learned by the transformer.
        :param num_heads: Number of attention heads in the transformer.
        :param num_layers: Number of transformer encoder layers.
        :param hidden_size: Size of hidden layers in the MLP head.
        :param dropout_prob: Dropout probability for regularization.
        """
        super(CategoricalTransformer, self).__init__()

        # Linear projection to embedding space since one-hot encoded data is already expanded
        self.embedding_layer = nn.Linear(input_size, embedding_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the CategoricalTransformer.

        :param x: One-hot encoded input tensor of shape [batch_size, num_features].
        :return: A tuple containing (hidden representations, final output).
        """
        # Embed one-hot encoded input (linear projection)
        x_embed = self.embedding_layer(x)  # Shape: [batch_size, embedding_dim]

        # Add sequence dimension expected by the transformer
        x_embed = x_embed.unsqueeze(1)  # Shape: [batch_size, 1, embedding_dim]

        # Transformer encoder
        x_transformed = self.transformer(x_embed)  # Shape: [batch_size, 1, embedding_dim]

        return x_transformed.squeeze()

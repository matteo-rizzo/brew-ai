import torch
from torch import nn


class FeatureEmbedding(nn.Module):
    def __init__(self, input_size: int, embedding_dim: int = 16):
        """
        FeatureEmbedding class that handles the optional embedding of input features.

        :param input_size: The size of the input features.
        :param embedding_dim: The dimension of the feature embeddings.
        """
        super(FeatureEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(input_size, embedding_dim)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feature embedding.

        :param x: The input tensor of shape [batch_size, input_size].
        :return: Embedded features if enabled, else the original input.
        """
        return self.embedding(x)  # Shape: [batch_size, embedding_dim]

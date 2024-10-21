from typing import Optional

import torch
from torch import nn

from src.classes.evaluation.periodicity.factories.ActivationFactory import ActivationFactory


class FeatureEmbedding(nn.Module):
    def __init__(
            self,
            input_size: int,
            embedding_dim: int = 16,
            num_layers: int = 3,
            activation: Optional[str] = 'ReLU'
    ) -> None:
        """
        FeatureEmbedding class that handles the optional embedding of input features with a configurable number of layers.

        :param input_size: The size of the input features.
        :param embedding_dim: The dimension of the feature embeddings.
        :param num_layers: The number of embedding layers to apply.
        :param activation: The activation function to use between layers.
        """
        super(FeatureEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

        layers = []
        in_dim = input_size

        # Get the activation function once
        activation_fn = None
        if activation:
            activation_fn = ActivationFactory.get_activation_function(activation)

        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, embedding_dim))
            if activation_fn and i < num_layers:
                layers.append(activation_fn)
            in_dim = embedding_dim  # Update input dimension for the next layer

        self.embedding = nn.Sequential(*layers)

    def get_embedding_dim(self) -> int:
        """
        Returns the dimension of the embeddings.

        :return: The embedding dimension.
        """
        return self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feature embedding.

        :param x: The input tensor of shape [batch_size, input_size].
        :return: Embedded features.
        """
        return self.embedding(x)  # Shape: [batch_size, embedding_dim]

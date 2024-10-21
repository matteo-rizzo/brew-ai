from typing import Optional

import torch
from torch import nn

from src.classes.evaluation.periodicity.embedding.FeatureEmbedding import FeatureEmbedding
from src.classes.evaluation.periodicity.factories.ActivationFactory import ActivationFactory


class DeepFeatureEmbedding(FeatureEmbedding):
    def __init__(self, input_size: int, embedding_dim: int = 16, num_layers: int = 3, hidden_size: int = 64,
                 activation: Optional[str] = 'ReLU', batch_norm: bool = True, dropout_prob: float = 0.2) -> None:
        """
        AdvancedFeatureEmbedding uses a deep neural network to create sophisticated embeddings for continuous features.

        :param input_size: Size of the input features.
        :param embedding_dim: Dimension of the output embeddings.
        :param num_layers: Number of hidden layers.
        :param hidden_size: Number of units in hidden layers.
        :param activation: Activation function to use.
        :param batch_norm: Whether to use batch normalization.
        :param dropout_prob: Dropout probability for regularization.
        """
        super().__init__(input_size, embedding_dim, num_layers, activation)
        layers = []
        in_dim = input_size

        activation_fn = ActivationFactory.get_activation_function(activation)

        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            if activation_fn:
                layers.append(activation_fn)
            layers.append(nn.Dropout(dropout_prob))
            in_dim = hidden_size  # Update input dimension for the next layer

        # Output layer to get the embedding
        layers.append(nn.Linear(hidden_size, embedding_dim))
        self.embedding = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

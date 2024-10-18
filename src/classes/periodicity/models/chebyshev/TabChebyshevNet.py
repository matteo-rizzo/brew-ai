import torch
from torch import nn

from src.classes.periodicity.models.chebyshev.ChebyshevLayer import ChebyshevLayer


class TabChebyshevNet(nn.Module):
    def __init__(self, continuous_input_size, categorical_input_size, num_chebyshev_terms,
                 hidden_size, num_layers=3, dropout_prob=0.2, batch_norm=True):
        """
        TabChebyshevNet that accepts one-hot encoded categorical features.

        :param continuous_input_size: Number of continuous input features.
        :param categorical_input_size: Total number of one-hot encoded categorical features.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        :param hidden_size: Size of hidden layers.
        :param num_layers: Number of hidden layers.
        :param dropout_prob: Dropout probability for regularization.
        :param batch_norm: Whether to use batch normalization.
        """
        super(TabChebyshevNet, self).__init__()

        self.continuous_input_size = continuous_input_size
        self.categorical_input_size = categorical_input_size

        # Chebyshev Layer for continuous features
        self.chebyshev_layer = ChebyshevLayer(continuous_input_size, num_chebyshev_terms, normalize=True)

        total_continuous_features = continuous_input_size * num_chebyshev_terms
        total_categorical_features = categorical_input_size  # One-hot encoded features

        total_features = total_continuous_features + total_categorical_features

        # Optionally, process categorical features through an MLP
        self.categorical_mlp = nn.Sequential(
            nn.Linear(categorical_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # Adjust total features if using MLP for categorical features
        total_categorical_features = hidden_size

        total_features = total_continuous_features + total_categorical_features

        # Fully Connected Layers with flexible depth
        layers = []
        input_size = total_features
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_prob))
            input_size = hidden_size  # Hidden layer size remains constant after the first layer

        # Output Layer
        layers.append(nn.Linear(hidden_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x_continuous, x_categorical):
        """
        Forward pass of the network.

        :param x_continuous: Tensor of continuous features, shape [batch_size, continuous_input_size].
        :param x_categorical: Tensor of one-hot encoded categorical features, shape [batch_size, categorical_input_size].
        """
        # Apply Chebyshev transformation to continuous features
        x_chebyshev = self.chebyshev_layer(x_continuous)  # Shape: [batch_size, total_continuous_features]

        # Process one-hot encoded categorical features through MLP
        x_categorical_processed = self.categorical_mlp(x_categorical)  # Shape: [batch_size, hidden_size]

        # Combine features
        x_combined = torch.cat([x_chebyshev, x_categorical_processed], dim=1)  # Shape: [batch_size, total_features]

        # Feed combined features into fully connected layers
        return self.network(x_combined)

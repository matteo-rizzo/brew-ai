import torch
from torch import nn

from src.classes.periodicity.models.chebyshev.ChebyshevLayer import ChebyshevLayer
from src.classes.periodicity.models.fourier.FourierLayer import FourierLayer


class TabPNPNet(nn.Module):
    def __init__(self, periodic_input_size, non_periodic_input_size, categorical_input_size,
                 num_fourier_features, num_chebyshev_terms,
                 hidden_size, num_layers=3, dropout_prob=0.2, batch_norm=True):
        """
        TabPNPNet extends PNPNet to support both continuous (periodic and non-periodic) and one-hot encoded categorical features.

        :param periodic_input_size: Number of periodic continuous input features.
        :param non_periodic_input_size: Number of non-periodic continuous input features.
        :param categorical_input_size: Total number of one-hot encoded categorical features.
        :param num_fourier_features: Number of Fourier features per periodic input feature.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        :param hidden_size: Number of neurons in hidden layers.
        :param num_layers: Number of hidden layers.
        :param dropout_prob: Dropout probability for regularization.
        :param batch_norm: Whether to use batch normalization.
        """
        super(TabPNPNet, self).__init__()
        self.fourier_layer = FourierLayer(periodic_input_size, num_fourier_features)
        self.chebyshev_layer = ChebyshevLayer(non_periodic_input_size, num_chebyshev_terms)

        # MLP for one-hot encoded categorical features
        self.categorical_mlp = nn.Sequential(
            nn.Linear(categorical_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # Calculate total features
        total_fourier_features = periodic_input_size * num_fourier_features * 2  # Times 2 for sin and cos
        total_chebyshev_features = non_periodic_input_size * num_chebyshev_terms
        total_categorical_features = hidden_size  # Output size of categorical MLP

        total_features = total_fourier_features + total_chebyshev_features + total_categorical_features

        # Build the network with batch normalization and dropout options
        layers = []
        input_size = total_features
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_prob))
            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(hidden_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x_periodic, x_non_periodic, x_categorical):
        # Apply Fourier and Chebyshev layers
        x_fourier = self.fourier_layer(x_periodic)
        x_chebyshev = self.chebyshev_layer(x_non_periodic)

        # Process one-hot encoded categorical features through MLP
        x_categorical_processed = self.categorical_mlp(x_categorical)

        # Concatenate Fourier, Chebyshev, and categorical outputs
        x_combined = torch.cat([x_fourier, x_chebyshev, x_categorical_processed], dim=-1)

        # Feed the combined features into the fully connected layers
        return self.network(x_combined)

import torch
from torch import nn

from src.classes.periodicity.models.chebyshev.ChebyshevLayer import ChebyshevLayer
from src.classes.periodicity.models.fourier.FourierLayer import FourierLayer


class PNPNet(nn.Module):
    def __init__(self, periodic_input_size, non_periodic_input_size, num_fourier_features, num_chebyshev_terms,
                 hidden_size, dropout_prob=0.2, batch_norm=True):
        """
        A combined neural network that integrates Fourier and Chebyshev layers.

        :param periodic_input_size: Size of periodic input features.
        :param non_periodic_input_size: Size of non-periodic input features.
        :param num_fourier_features: Number of Fourier features to learn.
        :param num_chebyshev_terms: Number of Chebyshev terms.
        :param hidden_size: Number of neurons in hidden layers.
        :param dropout_prob: Dropout probability for regularization.
        :param batch_norm: Whether to use batch normalization.
        """
        super(PNPNet, self).__init__()
        self.fourier_layer = FourierLayer(periodic_input_size, num_fourier_features)
        self.chebyshev_layer = ChebyshevLayer(non_periodic_input_size, num_chebyshev_terms)

        total_features = (periodic_input_size * num_fourier_features * 2) + (
                    non_periodic_input_size * num_chebyshev_terms)

        # Build the network with batch normalization and dropout options
        layers = [nn.Linear(total_features, hidden_size)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))  # Batch normalization
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout_prob))

        # Additional hidden layers
        layers.append(nn.Linear(hidden_size, hidden_size))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout_prob))

        # Output layer
        layers.append(nn.Linear(hidden_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x_periodic, x_non_periodic):
        # Apply Fourier and Chebyshev layers
        x_fourier = self.fourier_layer(x_periodic)
        x_chebyshev = self.chebyshev_layer(x_non_periodic)

        # Concatenate Fourier and Chebyshev outputs
        x_combined = torch.cat([x_fourier, x_chebyshev], dim=-1)

        # Feed the combined features into the fully connected layers
        return self.network(x_combined)

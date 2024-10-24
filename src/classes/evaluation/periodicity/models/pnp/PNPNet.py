import torch
from torch import nn

from src.classes.evaluation.periodicity.models.chebyshev.ChebyshevBlock import ChebyshevBlock
from src.classes.evaluation.periodicity.models.fourier.FourierBlock import FourierBlock
from src.classes.evaluation.periodicity.models.regressor.MLPRegressor import MLPRegressor


class PNPNet(nn.Module):
    def __init__(
            self,
            periodic_input_size: int,
            non_periodic_input_size: int,
            num_fourier_features: int,
            num_chebyshev_terms: int
    ):
        """
        A combined neural network that integrates Fourier and Chebyshev layers with an MLP regressor.
        Includes residual connections for both periodic and non-periodic inputs.

        :param periodic_input_size: Size of periodic input features.
        :param non_periodic_input_size: Size of non-periodic input features.
        :param num_fourier_features: Number of Fourier features to learn.
        :param num_chebyshev_terms: Number of Chebyshev terms.
        """
        super(PNPNet, self).__init__()

        # Fourier layer for periodic features
        self.fourier_layer = FourierBlock(periodic_input_size, num_fourier_features)

        # Chebyshev layer for non-periodic features
        self.chebyshev_layer = ChebyshevBlock(non_periodic_input_size, num_chebyshev_terms)

        # Total feature sizes
        total_fourier_features = periodic_input_size * num_fourier_features * 2  # Fourier has sine and cosine
        total_chebyshev_features = non_periodic_input_size * num_chebyshev_terms
        total_features = total_fourier_features + total_chebyshev_features

        # MLPRegressor for handling the fully connected layers
        self.regressor = MLPRegressor(total_features)

    def forward(self, x_periodic: torch.Tensor, x_non_periodic: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PNPNet with residual connections.

        :param x_periodic: Tensor of periodic features, shape [batch_size, periodic_input_size].
        :param x_non_periodic: Tensor of non-periodic features, shape [batch_size, non_periodic_input_size].
        :return: Output tensor of shape [batch_size, 1].
        """
        # Apply Fourier transformation to periodic features
        x_fourier = self.fourier_layer(x_periodic)  # Shape: [batch_size, total_fourier_features]

        # Apply Chebyshev transformation to non-periodic features
        x_chebyshev = self.chebyshev_layer(x_non_periodic)  # Shape: [batch_size, total_chebyshev_features]

        # Concatenate Fourier and Chebyshev outputs
        x_combined = torch.cat([x_fourier, x_chebyshev], dim=-1)  # Shape: [batch_size, total_features]

        # Pass through the MLP regressor
        out = self.regressor(x_combined)

        return out

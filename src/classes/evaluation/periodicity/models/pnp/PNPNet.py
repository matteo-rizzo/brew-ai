import torch
from torch import nn

from src.classes.evaluation.periodicity.models.chebyshev.ChebyshevLayer import ChebyshevLayer
from src.classes.evaluation.periodicity.models.fourier.FourierLayer import FourierLayer
from src.classes.evaluation.periodicity.models.mlp.MLPRegressor import MLPRegressor


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

        :param periodic_input_size: Size of periodic input features.
        :param non_periodic_input_size: Size of non-periodic input features.
        :param num_fourier_features: Number of Fourier features to learn.
        :param num_chebyshev_terms: Number of Chebyshev terms.
        """
        super(PNPNet, self).__init__()

        # Fourier and Chebyshev layers
        self.fourier_layer = FourierLayer(periodic_input_size, num_fourier_features)
        self.chebyshev_layer = ChebyshevLayer(non_periodic_input_size, num_chebyshev_terms)

        total_fourier_features = periodic_input_size * num_fourier_features * 2
        total_chebyshev_features = non_periodic_input_size * num_chebyshev_terms
        total_features = total_fourier_features + total_chebyshev_features

        # MLPRegressor for handling the fully connected layers
        self.mlp_regressor = MLPRegressor(total_features)

    def forward(self, x_periodic: torch.Tensor, x_non_periodic: torch.Tensor) -> torch.Tensor:
        # Apply Fourier transformation to periodic features
        x_fourier = self.fourier_layer(x_periodic)  # Shape: [batch_size, total_fourier_features]

        # Apply Chebyshev transformation to non-periodic features
        x_chebyshev = self.chebyshev_layer(x_non_periodic)  # Shape: [batch_size, total_chebyshev_features]

        # Concatenate Fourier and Chebyshev outputs
        x_combined = torch.cat([x_fourier, x_chebyshev], dim=-1)  # Shape: [batch_size, total_features]

        # Pass through the MLP regressor
        out = self.mlp_regressor(x_combined)

        return out

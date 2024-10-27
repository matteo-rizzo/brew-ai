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

        # Fourier layer for periodic features (if applicable)
        self.fourier_layer = FourierBlock(periodic_input_size, num_fourier_features) if periodic_input_size > 0 else None

        # Chebyshev layer for non-periodic features (if applicable)
        self.chebyshev_layer = ChebyshevBlock(non_periodic_input_size, num_chebyshev_terms) if non_periodic_input_size > 0 else None

        # Calculate total features dynamically based on the presence of Fourier and Chebyshev layers
        total_fourier_features = periodic_input_size * num_fourier_features * 2 if self.fourier_layer else 0
        total_chebyshev_features = non_periodic_input_size * num_chebyshev_terms if self.chebyshev_layer else 0
        total_features = total_fourier_features + total_chebyshev_features

        # MLPRegressor for handling the fully connected layers
        self.regressor = MLPRegressor(total_features)

    def forward(self, x_periodic: torch.Tensor = None, x_non_periodic: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of PNPNet with residual connections.

        :param x_periodic: Tensor of periodic features, shape [batch_size, periodic_input_size].
        :param x_non_periodic: Tensor of non-periodic features, shape [batch_size, non_periodic_input_size].
        :return: Output tensor of shape [batch_size, 1].
        """
        # Initialize an empty list to collect transformed features
        transformed_features = []

        # Apply Fourier transformation if periodic features are provided
        if self.fourier_layer and x_periodic is not None:
            x_fourier = self.fourier_layer(x_periodic)
            transformed_features.append(x_fourier)

        # Apply Chebyshev transformation if non-periodic features are provided
        if self.chebyshev_layer and x_non_periodic is not None:
            x_chebyshev = self.chebyshev_layer(x_non_periodic)
            transformed_features.append(x_chebyshev)

        # Concatenate all non-empty transformed features
        x_combined = torch.cat(transformed_features, dim=-1) if transformed_features else torch.empty(0, device=x_periodic.device if x_periodic is not None else x_non_periodic.device)

        # Pass through the MLP regressor
        out = self.regressor(x_combined)

        return out

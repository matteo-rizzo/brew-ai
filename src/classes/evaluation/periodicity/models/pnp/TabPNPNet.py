import torch
from torch import nn

from src.classes.evaluation.periodicity.models.categorical.CategoricalMLP import CategoricalMLP
from src.classes.evaluation.periodicity.models.chebyshev.ChebyshevBlock import ChebyshevBlock
from src.classes.evaluation.periodicity.models.fourier.FourierBlock import FourierBlock
from src.classes.evaluation.periodicity.models.regressor.MLPRegressor import MLPRegressor


class TabPNPNet(nn.Module):
    def __init__(
            self,
            periodic_input_size: int,
            non_periodic_input_size: int,
            categorical_input_size: int,
            num_fourier_features: int,
            num_chebyshev_terms: int,
            hidden_size: int
    ):
        """
        TabPNPNet extends PNPNet to support both continuous (periodic and non-periodic) and one-hot encoded categorical features,
        with residual connections.

        :param periodic_input_size: Number of periodic continuous input features.
        :param non_periodic_input_size: Number of non-periodic continuous input features.
        :param categorical_input_size: Total number of one-hot encoded categorical features.
        :param num_fourier_features: Number of Fourier features per periodic input feature.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        :param hidden_size: Number of neurons in hidden layers.
        """
        super(TabPNPNet, self).__init__()

        # Fourier and Chebyshev layers for periodic and non-periodic features
        self.fourier_layer = FourierBlock(periodic_input_size, num_fourier_features)
        self.chebyshev_layer = ChebyshevBlock(non_periodic_input_size, num_chebyshev_terms)

        # Residual connections for periodic and non-periodic features
        total_fourier_features = periodic_input_size * num_fourier_features * 2  # Times 2 for sin and cos
        total_chebyshev_features = non_periodic_input_size * num_chebyshev_terms

        # Categorical MLP for processing one-hot encoded categorical features
        self.categorical_layer = CategoricalMLP(categorical_input_size, hidden_size)

        # Total features after combining Fourier, Chebyshev, and categorical features
        total_categorical_features = hidden_size
        total_features = total_fourier_features + total_chebyshev_features + total_categorical_features

        # MLPRegressor for combined processing
        self.regressor = MLPRegressor(total_features)

    def forward(self, x_periodic, x_non_periodic, x_categorical):
        """
        Forward pass of the TabPNPNet with residual connections.

        :param x_periodic: Tensor of periodic features, shape [batch_size, periodic_input_size].
        :param x_non_periodic: Tensor of non-periodic features, shape [batch_size, non_periodic_input_size].
        :param x_categorical: Tensor of one-hot encoded categorical features, shape [batch_size, categorical_input_size].
        :return: Output tensor after passing through the network.
        """
        # Apply Fourier transformation to periodic features
        x_fourier = self.fourier_layer(x_periodic)

        # Apply Chebyshev transformation to non-periodic features
        x_chebyshev = self.chebyshev_layer(x_non_periodic)

        # Process one-hot encoded categorical features through MLP
        x_categorical_processed = self.categorical_layer(x_categorical)

        # Concatenate Fourier, Chebyshev, and categorical outputs
        x_combined = torch.cat([x_fourier, x_chebyshev, x_categorical_processed], dim=-1)

        # Pass the combined features through the regressor
        return self.regressor(x_combined)

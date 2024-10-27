import torch
from torch import nn

from src.classes.evaluation.periodicity.models.categorical.CategoricalMLP import CategoricalMLP
from src.classes.evaluation.periodicity.models.chebyshev.ChebyshevBlock import ChebyshevBlock
from src.classes.evaluation.periodicity.models.classifier.MLPClassifier import MLPClassifier
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
            hidden_size: int,
            output_size: int = 1
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
        :param output_size: Output size; if > 1, a classifier is used; otherwise, a regressor.
        """
        super(TabPNPNet, self).__init__()

        # Initialize Fourier layer if periodic_input_size > 0
        self.fourier_layer = FourierBlock(periodic_input_size,
                                          num_fourier_features) if periodic_input_size > 0 else None

        # Initialize Chebyshev layer if non_periodic_input_size > 0
        self.chebyshev_layer = ChebyshevBlock(non_periodic_input_size,
                                              num_chebyshev_terms) if non_periodic_input_size > 0 else None

        # Initialize categorical layer if categorical_input_size > 0
        self.categorical_layer = CategoricalMLP(categorical_input_size,
                                                hidden_size) if categorical_input_size > 0 else None

        # Compute the total feature sizes based on the presence of each layer
        total_fourier_features = periodic_input_size * num_fourier_features * 2 if self.fourier_layer else 0
        total_chebyshev_features = non_periodic_input_size * num_chebyshev_terms if self.chebyshev_layer else 0
        total_categorical_features = hidden_size if self.categorical_layer else 0
        total_features = total_fourier_features + total_chebyshev_features + total_categorical_features

        # Choose between MLPClassifier and MLPRegressor based on output_size
        if output_size > 1:
            self.mlp = MLPClassifier(input_size=total_features, output_size=output_size)
        else:
            self.mlp = MLPRegressor(input_size=total_features)

    def forward(self, x_periodic=None, x_non_periodic=None, x_categorical=None):
        """
        Forward pass of the TabPNPNet with residual connections.

        :param x_periodic: Tensor of periodic features, shape [batch_size, periodic_input_size].
        :param x_non_periodic: Tensor of non-periodic features, shape [batch_size, non_periodic_input_size].
        :param x_categorical: Tensor of one-hot encoded categorical features, shape [batch_size, categorical_input_size].
        :return: Output tensor after passing through the network.
        """
        # Initialize a list to collect transformed features
        transformed_features = []

        # Apply Fourier transformation if x_periodic is provided
        if self.fourier_layer and x_periodic is not None:
            x_fourier = self.fourier_layer(x_periodic)
            transformed_features.append(x_fourier)

        # Apply Chebyshev transformation if x_non_periodic is provided
        if self.chebyshev_layer and x_non_periodic is not None:
            x_chebyshev = self.chebyshev_layer(x_non_periodic)
            transformed_features.append(x_chebyshev)

        # Process categorical features through the MLP if x_categorical is provided
        if self.categorical_layer and x_categorical is not None:
            x_categorical_processed = self.categorical_layer(x_categorical)
            transformed_features.append(x_categorical_processed)

        # Concatenate all non-empty transformed features
        x_combined = torch.cat(transformed_features, dim=-1) if transformed_features else torch.empty(0,
                                                                                                      device=x_periodic.device if x_periodic is not None else (
                                                                                                          x_non_periodic.device if x_non_periodic is not None else x_categorical.device))

        # Pass the combined features through the regressor
        return self.mlp(x_combined)

import torch
from torch import nn

from src.classes.evaluation.periodicity.models.categorical.CategoricalTransformer import CategoricalTransformer
from src.classes.evaluation.periodicity.models.chebyshev.AdaptiveChebyshevLayer import AdaptiveChebyshevLayer
from src.classes.evaluation.periodicity.models.chebyshev.ChebyshevLayer import ChebyshevLayer
from src.classes.evaluation.periodicity.models.fourier.FourierLayer import FourierLayer
from src.classes.evaluation.periodicity.models.categorical.CategoricalMLP import CategoricalMLP
from src.classes.evaluation.periodicity.models.regressor.TabMLPRegressor import TabMLPRegressor


class TabPNPNet(nn.Module):
    def __init__(self, periodic_input_size: int, non_periodic_input_size: int, categorical_input_size: int,
                 num_fourier_features: int, num_chebyshev_terms: int, hidden_size: int):
        """
        TabPNPNet extends PNPNet to support both continuous (periodic and non-periodic) and one-hot encoded categorical features.

        :param periodic_input_size: Number of periodic continuous input features.
        :param non_periodic_input_size: Number of non-periodic continuous input features.
        :param categorical_input_size: Total number of one-hot encoded categorical features.
        :param num_fourier_features: Number of Fourier features per periodic input feature.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        :param hidden_size: Number of neurons in hidden layers.
        """
        super(TabPNPNet, self).__init__()
        self.fourier_layer = FourierLayer(periodic_input_size, num_fourier_features)
        self.chebyshev_layer = AdaptiveChebyshevLayer(non_periodic_input_size, num_chebyshev_terms)

        # Categorical MLP for processing one-hot encoded categorical features
        self.categorical_layer = CategoricalTransformer(categorical_input_size, hidden_size)

        # Calculate total features
        total_fourier_features = periodic_input_size * num_fourier_features * 2  # Times 2 for sin and cos
        total_chebyshev_features = non_periodic_input_size * num_chebyshev_terms
        total_categorical_features = hidden_size  # Output size of categorical MLP

        total_features = total_fourier_features + total_chebyshev_features + total_categorical_features

        # TabMLPRegressor for combined processing
        self.tab_mlp: TabMLPRegressor = TabMLPRegressor(total_features)

    def forward(self, x_periodic, x_non_periodic, x_categorical):
        # Apply Fourier and Chebyshev layers
        x_fourier = self.fourier_layer(x_periodic)
        x_chebyshev = self.chebyshev_layer(x_non_periodic)

        # Process one-hot encoded categorical features through MLP
        x_categorical_processed = self.categorical_layer(x_categorical)

        # Concatenate Fourier, Chebyshev, and categorical outputs
        x_combined = torch.cat([x_fourier, x_chebyshev, x_categorical_processed], dim=-1)

        # Feed the combined features into the fully connected layers
        return self.tab_mlp(x_combined)

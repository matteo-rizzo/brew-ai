import torch
from torch import nn

from src.classes.evaluation.periodicity.models.categorical.CategoricalMLP import CategoricalMLP
from src.classes.evaluation.periodicity.models.classifier.MLPClassifier import MLPClassifier
from src.classes.evaluation.periodicity.models.fourier.FourierBlock import FourierBlock
from src.classes.evaluation.periodicity.models.regressor.MLPRegressor import MLPRegressor


class TabFourierNet(nn.Module):
    def __init__(
            self,
            continuous_input_size: int,
            categorical_input_size: int,
            num_fourier_features: int,
            hidden_size: int,
            output_size: int = 1
    ):
        """
        TabFourierNet that supports both continuous and categorical features with a residual connection for the
        continuous features.

        :param continuous_input_size: Number of continuous input features.
        :param categorical_input_size: Number of one-hot encoded categorical features.
        :param num_fourier_features: Number of Fourier features to learn.
        :param hidden_size: Size of hidden layers for the categorical feature processing.
        :param output_size: Output size; if > 1, a classifier is used; otherwise, a regressor.
        """
        super().__init__()

        # Fourier layer for processing continuous features
        self.fourier_layer = FourierBlock(continuous_input_size, num_fourier_features)

        # Categorical MLP for processing one-hot encoded categorical features
        self.categorical_layer = CategoricalMLP(categorical_input_size, hidden_size)

        # Total features for continuous and categorical inputs
        total_continuous_features = continuous_input_size * num_fourier_features * 2  # Fourier has sine and cosine
        total_features = total_continuous_features + hidden_size

        # Residual layer for stability
        self.residual_layer = nn.Linear(continuous_input_size, 1)

        # Choose between MLPClassifier and MLPRegressor based on output_size
        if output_size > 1:
            self.mlp = MLPClassifier(input_size=total_features, output_size=output_size)
        else:
            self.mlp = MLPRegressor(input_size=total_features)

    def forward(self, x_continuous: torch.Tensor, x_categorical: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        :param x_continuous: Tensor of continuous features, shape [batch_size, continuous_input_size].
        :param x_categorical: Tensor of one-hot encoded categorical features, shape [batch_size, categorical_input_size].
        :return: Output tensor after passing through the network.
        """
        # Fourier transform continuous features
        x_fourier = self.fourier_layer(x_continuous)  # Shape: [batch_size, total_fourier_features]

        # Process one-hot encoded categorical features through the MLP
        x_categorical_processed = self.categorical_layer(x_categorical)  # Shape: [batch_size, hidden_size]

        # Combine continuous (Fourier + residual) and categorical features
        x_combined = torch.cat([x_fourier, x_categorical_processed],
                               dim=1)  # Shape: [batch_size, total_features]

        # Pass through the MLP regressor
        return self.mlp(x_combined)

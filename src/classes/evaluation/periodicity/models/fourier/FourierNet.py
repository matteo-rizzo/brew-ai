import torch
from torch import nn

from src.classes.evaluation.periodicity.models.fourier.FourierLayer import FourierLayer
from src.classes.evaluation.periodicity.models.mlp.MLPRegressor import MLPRegressor


class FourierNet(nn.Module):
    def __init__(self, input_size: int, num_fourier_features: int):
        """
        FourierNet integrates Fourier transformations with an MLP regressor.

        :param input_size: Size of the input features.
        :param num_fourier_features: Number of Fourier features to learn.
        """
        super(FourierNet, self).__init__()

        # Fourier Layer
        self.fourier_layer = FourierLayer(input_size, num_fourier_features)

        total_features = input_size * num_fourier_features * 2  # We have sine and cosine outputs

        # MLPRegressor for handling the fully connected layers
        self.mlp_regressor = MLPRegressor(total_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Fourier transformation to input features
        x_fourier = self.fourier_layer(x)

        # Pass through the MLP regressor
        out = self.mlp_regressor(x_fourier)

        return out

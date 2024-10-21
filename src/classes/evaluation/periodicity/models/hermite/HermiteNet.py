import torch.nn as nn

from src.classes.evaluation.periodicity.models.hermite.AdaptiveHermiteConvLayer import AdaptiveHermiteConvLayer
from src.classes.evaluation.periodicity.models.regressor.MLPRegressor import MLPRegressor


class HermiteNet(nn.Module):
    def __init__(self, input_size: int, hermite_degree: int):
        """
        Neural network based on Hermite polynomials.

        :param input_size: Number of input features.
        :param hermite_degree: Degree of Hermite polynomials to use.
        """
        super(HermiteNet, self).__init__()

        # Hermite polynomial layer
        self.hermite_layer = AdaptiveHermiteConvLayer(input_size, hermite_degree)

        total_features = input_size * (hermite_degree + 1)  # Size after Hermite layer

        self.regressor = MLPRegressor(total_features)

    def forward(self, x):
        """
        Forward pass of the Hermite-based neural network.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Output tensor of shape [batch_size, output_size].
        """
        # Apply Hermite polynomials to the input
        x = self.hermite_layer(x)

        # Pass through the MLP layers
        out = self.regressor(x)
        return out

import torch
from torch import nn

from src.classes.evaluation.periodicity.models.fourier.FourierEncoder import FourierEncoder
from src.classes.evaluation.periodicity.models.orthogonal_poly.OrthogonalPolynomialFactory import \
    OrthogonalPolynomialFactory
from src.config import POLY_TYPE


class AutoPNPEncoder(nn.Module):
    def __init__(self, input_size: int, num_fourier_features: int, max_poly_terms: int):
        """
        AutoPNPNet that integrates Fourier and Chebyshev layers.

        :param input_size: Size of the input features.
        :param num_fourier_features: Number of Fourier features to generate.
        :param max_poly_terms: Number of Chebyshev polynomial terms.
        """
        super(AutoPNPEncoder, self).__init__()

        # Fourier and Chebyshev layers
        self.fourier_layer = FourierEncoder(input_size, num_fourier_features)
        self.orthogonal_poly_layer = OrthogonalPolynomialFactory.get_polynomial(POLY_TYPE)(input_size, max_poly_terms)
        self.output_dim = self.fourier_layer.output_dim + self.orthogonal_poly_layer.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Fourier and Chebyshev transformations
        x_fourier = self.fourier_layer(x)
        x_orthogonal_poly = self.orthogonal_poly_layer(x)

        # Combine Fourier and Chebyshev transformed features
        return torch.cat([x_fourier, x_orthogonal_poly], dim=1)

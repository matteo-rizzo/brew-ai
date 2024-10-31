import torch
from torch import nn

from src.classes.evaluation.periodicity.models.chebyshev.ChebyshevEncoder import ChebyshevEncoder
from src.classes.evaluation.periodicity.models.fourier.FourierEncoder import FourierEncoder


class AutoPNPEncoder(nn.Module):
    def __init__(self, input_size: int, num_fourier_features: int, num_chebyshev_terms: int):
        """
        AutoPNPNet that integrates Fourier and Chebyshev layers.

        :param input_size: Size of the input features.
        :param num_fourier_features: Number of Fourier features to generate.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        """
        super(AutoPNPEncoder, self).__init__()

        # Fourier and Chebyshev layers
        self.fourier_layer = FourierEncoder(input_size, num_fourier_features)
        self.chebyshev_layer = ChebyshevEncoder(input_size, num_chebyshev_terms)
        self.output_dim = self.fourier_layer.output_dim + self.chebyshev_layer.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Fourier and Chebyshev transformations
        x_fourier = self.fourier_layer(x)
        x_chebyshev = self.chebyshev_layer(x)

        # Combine Fourier and Chebyshev transformed features
        return torch.cat([x_fourier, x_chebyshev], dim=1)

import torch
from torch import nn

from src.classes.evaluation.periodicity.models.autopnp.AutoPNPBlock import AutoPNPBlock
from src.classes.evaluation.periodicity.models.classifier.MLPClassifier import MLPClassifier
from src.classes.evaluation.periodicity.models.regressor.MLPRegressor import MLPRegressor


class AutoPNPNet(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_fourier_features: int,
            num_chebyshev_terms: int,
            num_layers: int = 1,
            compression_dim: int = 128,
            output_size: int = 1
    ):
        """
        AutoPNPNet integrates the AutoPNPBlock with Fourier features, Chebyshev terms, and an MLP regressor or classifier.
        Also includes a residual connection for stability.

        :param input_size: Size of the input features.
        :param num_fourier_features: Number of Fourier features to generate.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        :param num_layers: Number of AutoPNPEncoder layers to stack.
        :param compression_dim: Dimension to compress features to between layers.
        :param output_size: Output size; if > 1, a classifier is used; otherwise, a regressor.
        """
        super(AutoPNPNet, self).__init__()

        # Initialize AutoPNPBlock with Fourier and Chebyshev features
        self.autopnp_block = AutoPNPBlock(
            input_size=input_size,
            num_fourier_features=num_fourier_features,
            num_chebyshev_terms=num_chebyshev_terms,
            num_layers=num_layers,
            compression_dim=compression_dim
        )

        # Feature dimensions after the AutoPNPBlock transformations
        total_features = self.autopnp_block.output_dim

        # Choose between MLPClassifier and MLPRegressor based on output_size
        if output_size > 1:
            self.mlp = MLPClassifier(input_size=total_features, output_size=output_size)
        else:
            self.mlp = MLPRegressor(input_size=total_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for AutoPNPNet.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Output tensor of shape [batch_size, output_size].
        """
        # Pass input through the AutoPNPBlock
        x_combined = self.autopnp_block(x)

        # MLP (regressor or classifier) on the combined features
        return self.mlp(x_combined)

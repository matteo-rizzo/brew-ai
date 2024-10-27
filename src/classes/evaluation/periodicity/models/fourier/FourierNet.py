import torch
from torch import nn

from src.classes.evaluation.periodicity.models.classifier.MLPClassifier import MLPClassifier
from src.classes.evaluation.periodicity.models.fourier.FourierBlock import FourierBlock
from src.classes.evaluation.periodicity.models.regressor.MLPRegressor import MLPRegressor


class FourierNet(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_fourier_features: int,
            num_layers: int = 1,
            compression_dim: int = 128,
            dropout_prob: float = 0.2,
            output_size: int = 1
    ):
        """
        FourierNet integrates Fourier transformations with an MLP regressor and adds a residual connection.

        :param input_size: Size of the input features.
        :param num_fourier_features: Number of Fourier features to learn.
        :param num_layers: Number of FourierBlock layers to stack.
        :param compression_dim: Dimension to compress features to between layers.
        :param dropout_prob: Probability of dropout after normalization layers (0 to 1).
        :param output_size: Output size; if > 1, a classifier is used; otherwise, a regressor.
        """
        super(FourierNet, self).__init__()

        # Fourier Block
        self.fourier_block = FourierBlock(
            input_size=input_size,
            num_fourier_features=num_fourier_features,
            num_layers=num_layers,
            compression_dim=compression_dim,
            dropout_prob=dropout_prob
        )

        total_features = self.fourier_block.output_dim

        # Choose between MLPClassifier and MLPRegressor based on output_size
        if output_size > 1:
            self.mlp = MLPClassifier(input_size=total_features, output_size=output_size)
        else:
            self.mlp = MLPRegressor(input_size=total_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FourierNet with a residual connection.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Output tensor of shape [batch_size, 1].
        """
        # Apply Fourier transformation through the FourierBlock
        x_fourier = self.fourier_block(x)

        # Pass through the MLP regressor
        out = self.mlp(x_fourier)

        return out

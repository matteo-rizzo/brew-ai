import torch
from torch import nn

from src.classes.evaluation.periodicity.models.autopnp.AutoPNPBlock import AutoPNPBlock
from src.classes.evaluation.periodicity.models.categorical.CategoricalMLP import CategoricalMLP
from src.classes.evaluation.periodicity.models.classifier.MLPClassifier import MLPClassifier
from src.classes.evaluation.periodicity.models.regressor.MLPRegressor import MLPRegressor


class TabAutoPNPNet(nn.Module):
    def __init__(
            self,
            continuous_input_size: int,
            categorical_input_size: int,
            num_fourier_features: int,
            num_chebyshev_terms: int,
            hidden_size: int,
            output_size: int = 1
    ):
        """
        TabAutoPNPNet: Extends AutoPNPNet to support both continuous and one-hot encoded categorical features.

        :param continuous_input_size: Number of continuous input features.
        :param categorical_input_size: Total number of one-hot encoded categorical features.
        :param num_fourier_features: Number of Fourier features per continuous input feature.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        :param hidden_size: Size of hidden layers for categorical features processing.
        :param output_size: Output size; if > 1, a classifier is used; otherwise, a regressor.
        """
        super(TabAutoPNPNet, self).__init__()

        # Continuous feature processing using Fourier and Chebyshev polynomials
        self.autopnp_layer = AutoPNPBlock(
            input_size=continuous_input_size,
            num_fourier_features=num_fourier_features,
            num_chebyshev_terms=num_chebyshev_terms
        )

        # MLP for one-hot encoded categorical feature processing
        self.categorical_layer = CategoricalMLP(
            input_size=categorical_input_size,
            hidden_size=hidden_size
        )

        # Feature dimensions after the Fourier and Chebyshev transformations
        total_fourier_features = num_fourier_features * 2 * continuous_input_size  # Fourier has sine and cosine
        total_chebyshev_features = num_chebyshev_terms * continuous_input_size * 4

        # Total features after combining continuous and processed categorical features
        total_features = total_fourier_features + total_chebyshev_features + hidden_size

        # Choose between MLPClassifier and MLPRegressor based on output_size
        if output_size > 1:
            self.mlp = MLPClassifier(input_size=total_features, output_size=output_size)
        else:
            self.mlp = MLPRegressor(input_size=total_features)

        # Residual layer for continuous features (without categorical)
        self.residual_layer = nn.Linear(total_fourier_features + total_chebyshev_features, 1)

    def forward(self, x_continuous: torch.Tensor, x_categorical: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TabAutoPNPNet.

        :param x_continuous: Continuous input tensor of shape [batch_size, continuous_input_size].
        :param x_categorical: Categorical input tensor of shape [batch_size, categorical_input_size].
        :return: Output tensor of shape [batch_size, 1].
        """
        # Process continuous features through AutoPNPBlock (Fourier + Chebyshev)
        x_continuous_combined = self.autopnp_layer(x_continuous)

        # Residual connection applied to continuous features
        residual = self.residual_layer(x_continuous_combined)

        # Process one-hot encoded categorical features through MLP
        x_categorical_processed = self.categorical_layer(x_categorical)  # Shape: [batch_size, hidden_size]

        # Combine continuous and categorical processed features
        x_combined = torch.cat([x_continuous_combined, x_categorical_processed], dim=1)

        # Pass combined features through the regressor MLP
        out = self.mlp(x_combined)

        # Add the residual (continuous features only)
        out += residual

        return out

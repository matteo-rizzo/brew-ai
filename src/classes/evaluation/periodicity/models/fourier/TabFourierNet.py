import torch
from torch import nn

from src.classes.evaluation.periodicity.models.fourier.FourierLayer import FourierLayer
from src.classes.evaluation.periodicity.models.mlp.CategoricalMLP import CategoricalMLP
from src.classes.evaluation.periodicity.models.mlp.TabMLP import TabMLP


class TabFourierNet(nn.Module):
    def __init__(
            self,
            continuous_input_size: int,
            categorical_input_size: int,
            num_fourier_features: int,
            hidden_size: int
    ):
        """
        TabFourierNet that supports both continuous and categorical features.

        :param continuous_input_size: Number of continuous input features.
        :param categorical_input_size: Number of one-hot encoded categorical features.
        :param num_fourier_features: Number of Fourier features to learn.
        :param hidden_size: Size of hidden layers.
        :param num_layers: Number of hidden layers in the MLP.
        :param dropout_prob: Dropout probability for regularization.
        :param batch_norm: Whether to use batch normalization.
        """
        super().__init__()

        # Fourier layer for processing continuous features
        self.fourier_layer = FourierLayer(continuous_input_size, num_fourier_features)

        # Categorical MLP for processing one-hot encoded categorical features
        self.categorical_mlp = CategoricalMLP(categorical_input_size, hidden_size)

        # Update total feature size (continuous + categorical)
        total_continuous_features = continuous_input_size * num_fourier_features * 2
        total_features = total_continuous_features + hidden_size  # Hidden size of processed categorical features

        # TabMLP for combined processing
        self.tab_mlp: TabMLP = TabMLP(total_features)

    def forward(self, x_continuous: torch.Tensor, x_categorical: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        :param x_continuous: Tensor of continuous features, shape [batch_size, continuous_input_size].
        :param x_categorical: Tensor of one-hot encoded categorical features, shape [batch_size, categorical_input_size].
        :return: Output tensor after passing through the network.
        """
        # Fourier transform continuous features
        x_fourier = self.fourier_layer(x_continuous)  # Shape: [batch_size, total_fourier_features]

        # Process one-hot encoded categorical features through MLP
        x_categorical_processed = self.categorical_mlp(x_categorical)  # Shape: [batch_size, hidden_size]

        # Combine Fourier and categorical features
        x_combined = torch.cat([x_fourier, x_categorical_processed], dim=1)  # Shape: [batch_size, total_features]

        # Pass through the MLP regressor
        return self.tab_mlp(x_combined)

import torch
from torch import Tensor, nn

from src.classes.evaluation.periodicity.models.chebyshev.ChebyshevLayer import ChebyshevLayer
from src.classes.evaluation.periodicity.models.mlp.CategoricalMLP import CategoricalMLP
from src.classes.evaluation.periodicity.models.mlp.TabMLP import TabMLP


class TabChebyshevNet(nn.Module):
    def __init__(
            self,
            continuous_input_size: int,
            categorical_input_size: int,
            num_chebyshev_terms: int,
            hidden_size: int
    ):
        """
        TabChebyshevNet that accepts one-hot encoded categorical features.

        :param continuous_input_size: Number of continuous input features.
        :param categorical_input_size: Total number of one-hot encoded categorical features.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        :param hidden_size: Size of hidden layers.
        """
        super().__init__()

        # Chebyshev layer for processing continuous features
        self.chebyshev_layer = ChebyshevLayer(continuous_input_size, num_chebyshev_terms, normalize=True)

        # MLP for processing categorical features
        self.categorical_mlp: CategoricalMLP = CategoricalMLP(categorical_input_size, hidden_size)

        # Update total feature size (continuous + categorical)
        total_continuous_features: int = continuous_input_size * num_chebyshev_terms
        total_features: int = total_continuous_features + hidden_size  # Hidden size of processed categorical features

        # TabMLP for combined processing
        self.tab_mlp: TabMLP = TabMLP(total_features)

    def forward(self, x_continuous: Tensor, x_categorical: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass of the network.

        :param x_continuous: Tensor of continuous features, shape [batch_size, continuous_input_size].
        :param x_categorical: Tensor of one-hot encoded categorical features, shape [batch_size, categorical_input_size].
        :return: Output tensor after passing through the network.
        """
        # Use ChebyshevNet's forward method for continuous features
        x_chebyshev: Tensor = self.chebyshev_layer(x_continuous)  # Shape: [batch_size, total_continuous_features]

        # Process categorical features
        x_categorical_processed: Tensor = self.categorical_mlp(x_categorical)  # Shape: [batch_size, hidden_size]

        # Combine continuous (Chebyshev) and categorical features - Shape: [batch_size, total_features]
        x_combined: Tensor = torch.cat([x_chebyshev, x_categorical_processed], dim=1)

        # Pass combined features through the TabMLP
        return self.tab_mlp(x_combined)

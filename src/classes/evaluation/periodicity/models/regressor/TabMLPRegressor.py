import torch
from torch import nn

from src.classes.evaluation.periodicity.factories.ActivationFactory import ActivationFactory


class TabMLPRegressor(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout_prob: float = 0.2,
            batch_norm: bool = True,
            activation: str = 'ReLU',
            output_size: int = 1
    ):
        """
        TabMLPRegressor class to handle the combined processing of continuous and categorical features.

        :param input_size: Total number of input features (continuous + categorical).
        :param hidden_size: Size of the hidden layers.
        :param num_layers: Number of layers in the MLP.
        :param dropout_prob: Dropout probability for regularization.
        :param batch_norm: Whether to apply batch normalization.
        :param activation: The activation function to apply between layers ('ReLU', 'SiLU', etc.).
        :param output_size: Size of the output layer (default 1 for regression).
        """
        super(TabMLPRegressor, self).__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(ActivationFactory().get_activation_function(activation))
            layers.append(nn.Dropout(dropout_prob))
            input_size = hidden_size  # The next layer's input size is the previous layer's output

        layers.append(nn.Linear(hidden_size, output_size))  # Final output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x_combined: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the combined continuous and categorical features.

        :param x_combined: Concatenated continuous and categorical feature tensor.
        :return: Final output of the model.
        """
        return self.network(x_combined)

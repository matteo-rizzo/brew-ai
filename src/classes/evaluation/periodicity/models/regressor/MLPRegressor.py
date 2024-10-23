import torch
from torch import nn

from src.classes.evaluation.periodicity.factories.ActivationFactory import ActivationFactory


class MLPRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, output_size: int = 1, num_layers: int = 1,
                 activation: str = 'ReLU', dropout_prob: float = 0.2, batch_norm: bool = True):
        """
        MLPRegressor: A flexible Multi-Layer Perceptron regressor module.

        :param input_size: Size of the input features.
        :param hidden_size: Size of hidden layers.
        :param output_size: Size of the output (1 for regression tasks).
        :param num_layers: Number of fully connected layers in the MLP.
        :param activation: Activation function to use ('ReLU', 'SiLU', 'LeakyReLU', etc.).
        :param dropout_prob: Dropout probability for regularization.
        :param batch_norm: Whether to use batch normalization.
        """
        super(MLPRegressor, self).__init__()

        # Store parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.batch_norm = batch_norm

        # Select activation function
        activation_fn = ActivationFactory.get_activation_function(activation)

        # Build the network layers
        layers = []
        current_input_size = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(current_input_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout_prob))
            current_input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        # Create the sequential network
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLPRegressor.
        :param x: Input tensor.
        :return: Output tensor (regression predictions).
        """
        return self.network(x)

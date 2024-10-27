import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int = 32,
            output_size: int = 1,
            num_layers: int = 1,
            dropout_prob: float = 0.2,
            batch_norm: bool = True,
            activation_fn: nn.Module = nn.ReLU()
    ):
        """
        MLPClassifier: A flexible Multi-Layer Perceptron classifier module.

        :param input_size: Size of the input features.
        :param hidden_size: Size of hidden layers.
        :param output_size: Number of classes for classification tasks.
        :param num_layers: Number of fully connected hidden layers in the MLP.
        :param dropout_prob: Dropout probability for regularization.
        :param batch_norm: Whether to use batch normalization.
        :param activation_fn: Activation function to apply after each layer (default is ReLU).
        :param output_activation: Activation function for the output layer (default is Softmax for classification).
        """
        super(MLPClassifier, self).__init__()

        # Build the network layers
        layers = []
        current_input_size = input_size

        if num_layers > 0:
            for _ in range(num_layers):
                layers.append(nn.Linear(current_input_size, hidden_size))

                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_size))

                layers.append(activation_fn)
                layers.append(nn.Dropout(dropout_prob))

                current_input_size = hidden_size

        # Output layer with output activation for classification
        layers.append(nn.Linear(current_input_size, output_size))

        # Create the sequential network
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLPClassifier.
        :param x: Input tensor of shape [batch_size, input_size].
        :return: Output tensor of shape [batch_size, output_size] (classification probabilities).
        """
        return self.network(x)

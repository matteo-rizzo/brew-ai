from torch import nn
from src.classes.evaluation.periodicity.factories.ActivationFactory import ActivationFactory
from src.classes.evaluation.periodicity.feature_selector.FeatureSelector import FeatureSelector


class PeriodicityFeatureSelector(FeatureSelector):
    def __init__(self, input_size, hidden_size=256, num_layers=3, activation='ReLU', dropout_prob=0.2,
                 use_batch_norm=True):
        """
        Enhanced PeriodicityFeatureSelector with additional layers, dropout, and optional batch normalization.

        :param input_size: Number of input features.
        :param hidden_size: Number of neurons in the hidden layers.
        :param num_layers: Number of hidden layers.
        :param activation: Activation function to use ('ReLU', 'LeakyReLU', 'SiLU', etc.).
        :param dropout_prob: Dropout probability for regularization.
        :param use_batch_norm: Whether to apply batch normalization after each layer.
        """
        super(PeriodicityFeatureSelector, self).__init__()

        dropout_prob = dropout_prob
        activation_fn = ActivationFactory.get_activation_function(activation)

        layers = []
        in_features = input_size

        # Hidden layers with optional batch normalization and dropout
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))  # Use Batch Normalization
            layers.append(activation_fn)
            if dropout_prob > 0:
                layers.append(nn.Dropout(p=dropout_prob))
            in_features = hidden_size

        # Final layer to map to periodicity scores
        layers.append(nn.Linear(hidden_size, input_size))
        layers.append(nn.Sigmoid())  # Produces periodicity scores between 0 and 1

        # Sequential model of all layers
        self.periodicity_estimator = nn.Sequential(*layers)

        # Apply Xavier initialization for all layers
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass to estimate the periodicity of each input feature.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Periodicity scores of shape [batch_size, input_size].
        """
        periodicity_scores = self.periodicity_estimator(x)  # Shape: [batch_size, input_size]
        return periodicity_scores

    def _initialize_weights(self):
        """
        Apply Xavier initialization to all linear layers.
        """
        for module in self.periodicity_estimator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

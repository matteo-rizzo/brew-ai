from torch import nn

from src.classes.evaluation.periodicity.feature_selector.FeatureSelector import FeatureSelector


class PeriodicityFeatureSelector(FeatureSelector):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        """
        Enhanced PeriodicityFeatureSelector with additional layers and an attention mechanism.

        :param input_size: Number of input features.
        :param hidden_size: Number of neurons in the hidden layers.
        :param num_layers: Number of hidden layers.
        """
        super(PeriodicityFeatureSelector, self).__init__()
        layers = []
        in_features = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.SiLU())
            in_features = hidden_size
        layers.append(nn.Linear(hidden_size, input_size))
        layers.append(nn.Sigmoid())  # Produces periodicity scores between 0 and 1
        self.periodicity_estimator = nn.Sequential(*layers)

    def forward(self, x):
        # Estimate the periodicity of each input feature
        periodicity_scores = self.periodicity_estimator(x)  # Shape: [batch_size, input_size]
        return periodicity_scores

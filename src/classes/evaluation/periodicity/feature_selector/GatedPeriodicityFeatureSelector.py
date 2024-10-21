from torch import nn

from src.classes.evaluation.periodicity.feature_selector.FeatureSelector import FeatureSelector


class GatedPeriodicityFeatureSelector(FeatureSelector):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        """
        Enhanced PeriodicityFeatureSelector using gating mechanisms.

        :param input_size: Number of input features.
        :param hidden_size: Number of neurons in the hidden layers.
        :param num_layers: Number of hidden layers.
        """
        super(GatedPeriodicityFeatureSelector, self).__init__()

        # Feature transformation layers
        self.feature_transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU()
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch_size, input_size]

        # Feature transformation
        x_transformed = self.feature_transform(x)  # [batch_size, hidden_size]

        # Gating
        gates = self.gate(x)  # [batch_size, input_size]

        # Apply gates to input features
        x_weighted = x * gates  # [batch_size, input_size]

        # Output layer
        periodicity_scores = self.output_layer(x_transformed)  # [batch_size, input_size]

        return periodicity_scores

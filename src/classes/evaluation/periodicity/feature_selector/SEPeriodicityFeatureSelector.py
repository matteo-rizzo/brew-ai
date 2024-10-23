from torch import nn

from src.classes.evaluation.periodicity.feature_selector.FeatureSelector import FeatureSelector


class SEPeriodicityFeatureSelector(FeatureSelector):
    def __init__(self, input_size, hidden_size=32, reduction_ratio=2, num_layers=1):
        """
        Enhanced PeriodicityFeatureSelector using Squeeze-and-Excitation blocks.

        :param input_size: Number of input features.
        :param hidden_size: Number of neurons in the hidden layers.
        :param reduction_ratio: Reduction ratio for SE blocks.
        :param num_layers: Number of hidden layers.
        """
        super(SEPeriodicityFeatureSelector, self).__init__()

        # Initial layers
        layers = []
        in_features = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        # Feature transformation
        self.feature_transform = nn.Sequential(*layers)

        # Squeeze-and-Excitation block
        self.se_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // reduction_ratio),
            nn.ReLU(),
            nn.Linear(hidden_size // reduction_ratio, hidden_size),
            nn.Sigmoid()
        )

        # Output layer to produce periodicity scores
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()  # Produces periodicity scores between 0 and 1
        )

    def forward(self, x):
        # x: [batch_size, input_size]

        # Initial transformation
        x_transformed = self.feature_transform(x)  # [batch_size, hidden_size]

        # Squeeze: Global information embedding
        x_squeezed = x_transformed.mean(dim=0)  # [hidden_size]

        # Excitation: Adaptive recalibration
        weights = self.se_block(x_squeezed)  # [hidden_size]

        # Recalibrate features
        x_recalibrated = x_transformed * weights  # [batch_size, hidden_size]

        # Output layer
        periodicity_scores = self.output_layer(x_recalibrated)  # [batch_size, input_size]

        return periodicity_scores

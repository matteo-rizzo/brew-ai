import torch
from torch import nn

from src.classes.periodicity.models.chebyshev.ChebyshevLayer import ChebyshevLayer
from src.classes.periodicity.models.fourier.FourierLayer import FourierLayer
from src.classes.periodicity.feature_selector.PeriodicityFeatureSelector import PeriodicityFeatureSelector


class TabAutoPNPNet(nn.Module):
    def __init__(self, continuous_input_size, categorical_input_size,
                 num_fourier_features, num_chebyshev_terms, hidden_size, num_layers=3,
                 dropout_prob=0.2, batch_norm=True, activation='SiLU'):
        """
        TabAutoPNPNet: Extends AutoPNPNet to support both continuous and one-hot encoded categorical features.

        :param continuous_input_size: Number of continuous input features.
        :param categorical_input_size: Total number of one-hot encoded categorical features.
        :param num_fourier_features: Number of Fourier features per continuous input feature.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        :param hidden_size: Size of hidden layers.
        :param num_layers: Number of hidden layers.
        :param dropout_prob: Dropout probability for regularization.
        :param batch_norm: Whether to use batch normalization.
        :param activation: Activation function to use in hidden layers.
        """
        super(TabAutoPNPNet, self).__init__()

        self.continuous_input_size = continuous_input_size
        self.categorical_input_size = categorical_input_size

        # Fourier and Chebyshev layers for continuous features
        self.fourier_layer = FourierLayer(continuous_input_size, num_fourier_features)
        self.chebyshev_layer = ChebyshevLayer(continuous_input_size, num_chebyshev_terms)

        self.total_fourier_features = continuous_input_size * num_fourier_features * 2  # Times 2 for sin and cos
        self.total_chebyshev_features = continuous_input_size * num_chebyshev_terms

        # Periodicity-based feature selector for continuous features
        self.feature_selector = PeriodicityFeatureSelector(continuous_input_size)

        # MLP for one-hot encoded categorical features
        self.categorical_mlp = nn.Sequential(
            nn.Linear(categorical_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # Total features after combining continuous and categorical features
        total_features = self.total_fourier_features + self.total_chebyshev_features + hidden_size

        activation_fn = self._get_activation_function(activation)

        # Fully Connected Layers
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(total_features, hidden_size),
                nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity(),
                activation_fn,
                nn.Dropout(dropout_prob)
            ])
            total_features = hidden_size

        # Output Layer
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

        # Residual connection layer for continuous features
        self.residual_layer = nn.Linear(self.total_fourier_features + self.total_chebyshev_features, 1)

    @staticmethod
    def _get_activation_function(name):
        if name == 'ReLU':
            return nn.ReLU()
        elif name == 'SiLU':
            return nn.SiLU()
        elif name == 'LeakyReLU':
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, x_continuous, x_categorical):
        batch_size = x_continuous.size(0)

        # Estimate periodicity scores for continuous features
        periodicity_scores = self.feature_selector(x_continuous)  # Shape: [batch_size, continuous_input_size]

        # Apply Fourier and Chebyshev transformations to continuous features
        x_fourier = self.fourier_layer(x_continuous)  # Shape: [batch_size, total_fourier_features]
        x_chebyshev = self.chebyshev_layer(x_continuous)  # Shape: [batch_size, total_chebyshev_features]

        # Reshape periodicity scores for gating
        num_fourier_features_per_input = self.fourier_layer.num_features_per_input * 2  # Times 2 for sin and cos
        num_chebyshev_features_per_input = self.chebyshev_layer.num_terms

        # Gating for Fourier features
        periodicity_scores_fourier = periodicity_scores.unsqueeze(2).repeat(1, 1, num_fourier_features_per_input)
        periodicity_scores_fourier = periodicity_scores_fourier.view(batch_size, -1)

        # Gating for Chebyshev features
        periodicity_scores_chebyshev = (1 - periodicity_scores).unsqueeze(2).repeat(1, 1,
                                                                                    num_chebyshev_features_per_input)
        periodicity_scores_chebyshev = periodicity_scores_chebyshev.view(batch_size, -1)

        # Apply gating
        x_fourier_weighted = x_fourier * periodicity_scores_fourier
        x_chebyshev_weighted = x_chebyshev * periodicity_scores_chebyshev

        # Combine continuous features
        x_continuous_combined = torch.cat([x_fourier_weighted, x_chebyshev_weighted], dim=1)

        # Residual connection for continuous features
        residual = self.residual_layer(x_continuous_combined)

        # Process one-hot encoded categorical features through MLP
        x_categorical_processed = self.categorical_mlp(x_categorical)  # Shape: [batch_size, hidden_size]

        # Combine continuous and categorical features
        x_combined = torch.cat([x_continuous_combined, x_categorical_processed], dim=1)

        # Pass through fully connected layers
        out = self.network(x_combined)

        # Add residual (only from continuous features)
        out += residual

        return out

import torch
from torch import nn

from src.classes.periodicity.models.chebyshev.ChebyshevLayer import ChebyshevLayer
from src.classes.periodicity.models.fourier.FourierLayer import FourierLayer
from src.classes.periodicity.feature_selector.PeriodicityFeatureSelector import PeriodicityFeatureSelector


class AutoPNPNet(nn.Module):
    def __init__(self, input_size, num_fourier_features, num_chebyshev_terms, hidden_size, num_layers=3,
                 dropout_prob=0.2, batch_norm=True, activation='SiLU'):
        super(AutoPNPNet, self).__init__()

        self.input_size = input_size
        self.fourier_layer = FourierLayer(input_size, num_fourier_features)
        self.chebyshev_layer = ChebyshevLayer(input_size, num_chebyshev_terms)

        self.total_fourier_features = input_size * num_fourier_features * 2
        self.total_chebyshev_features = input_size * num_chebyshev_terms
        total_features = self.total_fourier_features + self.total_chebyshev_features

        self.feature_selector = PeriodicityFeatureSelector(input_size)

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

        # Residual connection layer
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

    def forward(self, x):
        batch_size = x.size(0)
        # Estimate periodicity scores
        periodicity_scores = self.feature_selector(x)  # Shape: [batch_size, input_size]

        # Apply Fourier and Chebyshev transformations
        x_fourier = self.fourier_layer(x)  # Shape: [batch_size, total_fourier_features]
        x_chebyshev = self.chebyshev_layer(x)  # Shape: [batch_size, total_chebyshev_features]

        # Reshape periodicity scores for gating
        num_fourier_features_per_input = self.fourier_layer.num_features_per_input * 2
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

        # Combine features
        x_combined = torch.cat([x_fourier_weighted, x_chebyshev_weighted], dim=1)

        # Residual connection
        residual = self.residual_layer(x_combined)

        # Pass through fully connected layers
        out = self.network(x_combined)

        # Add residual
        out += residual

        return out
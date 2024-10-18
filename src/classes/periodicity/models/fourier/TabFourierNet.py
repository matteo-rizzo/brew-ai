import torch
from torch import nn

from src.classes.periodicity.models.fourier.FourierLayer import FourierLayer


# TabFourierNet that supports both continuous and categorical features
class TabFourierNet(nn.Module):
    def __init__(self, continuous_input_size, categorical_input_size, num_fourier_features, hidden_size,
                 num_layers=3, dropout_prob=0.2, batch_norm=True):
        super(TabFourierNet, self).__init__()

        self.continuous_input_size = continuous_input_size
        self.categorical_input_size = categorical_input_size

        # Fourier layer for continuous features
        self.fourier_layer = FourierLayer(continuous_input_size, num_fourier_features)

        # MLP for one-hot encoded categorical features
        self.categorical_mlp = nn.Sequential(
            nn.Linear(categorical_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # Adjust total features
        total_continuous_features = continuous_input_size * num_fourier_features * 2
        total_features = total_continuous_features + hidden_size  # Output size of categorical MLP

        # Fully connected layers
        layers = []
        input_size = total_features
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_prob))
            input_size = hidden_size

        # Output Layer
        layers.append(nn.Linear(hidden_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x_continuous, x_categorical_onehot):
        # Fourier transform continuous features
        x_fourier = self.fourier_layer(x_continuous)

        # Process one-hot encoded categorical features through MLP
        x_categorical_processed = self.categorical_mlp(x_categorical_onehot)

        # Combine features
        x_combined = torch.cat([x_fourier, x_categorical_processed], dim=1)

        # Pass through fully connected layers
        return self.network(x_combined)

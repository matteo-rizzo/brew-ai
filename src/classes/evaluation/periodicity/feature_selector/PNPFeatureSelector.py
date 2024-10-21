import torch
from torch import nn
import numpy as np

from src.classes.evaluation.periodicity.feature_selector.FeatureSelector import FeatureSelector


class PNPFeatureSelector(FeatureSelector):
    def __init__(self, input_size: int, num_fourier_features: int = 15, max_chebyshev_terms: int = 3,
                 selection_activation: str = 'sigmoid', init_frequency_range=(1.0, 3.0), normalize: bool = True):
        """
        Mixed Fourier-Chebyshev Feature Selector for periodic and non-periodic features.

        :param input_size: Number of input features.
        :param num_fourier_features: Number of Fourier features per input feature.
        :param max_chebyshev_terms: Maximum number of Chebyshev polynomial terms.
        :param selection_activation: Activation function for feature selection ('sigmoid' or 'softmax').
        :param init_frequency_range: Range for initializing Fourier frequencies.
        :param normalize: Whether to normalize the combined features.
        """
        super(PNPFeatureSelector, self).__init__()

        self.input_size = input_size
        self.num_fourier_features = num_fourier_features
        self.max_chebyshev_terms = max_chebyshev_terms
        self.normalize = normalize

        # Learnable frequency parameters for Fourier features
        self.B = nn.Parameter(torch.empty(input_size, num_fourier_features))
        nn.init.uniform_(self.B, *init_frequency_range)

        # Learnable weights for Chebyshev polynomial terms
        self.chebyshev_weights = nn.Parameter(torch.ones(input_size, max_chebyshev_terms))

        # Feature selector (sigmoid or softmax) to select between Fourier and Chebyshev features
        self.fc = nn.Linear(input_size * (num_fourier_features * 2 + max_chebyshev_terms), input_size)

        if selection_activation == 'sigmoid':
            self.selection_activation = nn.Sigmoid()
        elif selection_activation == 'softmax':
            self.selection_activation = nn.Softmax(dim=1)
        else:
            raise ValueError("Invalid selection_activation. Use 'sigmoid' or 'softmax'.")

        # Optional normalization layer for the combined features
        if self.normalize:
            self.norm_layer = nn.LayerNorm(input_size * (num_fourier_features * 2 + max_chebyshev_terms))

    def fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate Fourier features for each input feature.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Fourier features of shape [batch_size, input_size, num_fourier_features * 2].
        """
        x = x.unsqueeze(2)  # Shape: [batch_size, input_size, 1]
        B = self.B.unsqueeze(0)  # Shape: [1, input_size, num_fourier_features]
        x_proj = 2 * np.pi * x * B  # Shape: [batch_size, input_size, num_fourier_features]

        x_sin = torch.sin(x_proj)
        x_cos = torch.cos(x_proj)

        # Concatenate sin and cos features
        return torch.cat([x_sin, x_cos], dim=2)  # Shape: [batch_size, input_size, num_fourier_features * 2]

    def chebyshev_polynomials(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate Chebyshev polynomials for the input.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Chebyshev polynomials of shape [batch_size, input_size, max_chebyshev_terms].
        """
        x = x.unsqueeze(2)  # Shape: [batch_size, input_size, 1]
        T = [torch.ones_like(x), x]  # T0 and T1

        for _ in range(2, self.max_chebyshev_terms):
            T_next = 2 * x * T[-1] - T[-2]
            T.append(T_next)

        # Stack the Chebyshev polynomials
        x_cheb = torch.cat(T, dim=2)  # Shape: [batch_size, input_size, max_chebyshev_terms]

        # Weight the Chebyshev terms using learnable weights
        return x_cheb * self.chebyshev_weights.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Mixed Fourier-Chebyshev Feature Selector.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Selected features of shape [batch_size, input_size].
        """
        # Generate Fourier and Chebyshev features
        x_fourier = self.fourier_features(x)  # Shape: [batch_size, input_size, num_fourier_features * 2]
        x_chebyshev = self.chebyshev_polynomials(x)  # Shape: [batch_size, input_size, max_chebyshev_terms]

        # Flatten the Fourier and Chebyshev features
        x_combined = torch.cat([x_fourier, x_chebyshev], dim=2)  # Shape: [batch_size, input_size, total_features]
        x_combined = x_combined.view(x.size(0), -1)  # Shape: [batch_size, total_features_flattened]

        # Optionally normalize the combined features
        if self.normalize:
            x_combined = self.norm_layer(x_combined)

        # Apply the fully connected layer for feature selection
        x_selected = self.fc(x_combined)  # Shape: [batch_size, input_size]

        # Apply the selection activation (sigmoid or softmax)
        return self.selection_activation(x_selected)

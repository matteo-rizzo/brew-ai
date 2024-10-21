import numpy as np
import torch
from torch import nn


class FourierLayer(nn.Module):
    def __init__(self, input_size, num_features_per_input, scale=True, init_frequency_range=(1.0, 10.0)):
        """
        Optimized Learnable Fourier Transform Layer with per-feature transformations.

        :param input_size: Number of input features.
        :param num_features_per_input: Number of Fourier features per input feature.
        :param scale: Whether to scale the input before applying the Fourier Transform.
        :param init_frequency_range: Range for the initialization of the frequency matrix B.
        """
        super(FourierLayer, self).__init__()
        self.scale = scale
        self.input_size = input_size
        self.num_features_per_input = num_features_per_input

        # Per-feature frequency parameters
        self.B = nn.Parameter(torch.empty(input_size, num_features_per_input))
        nn.init.uniform_(self.B, *init_frequency_range)

        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(input_size))  # Per-feature scaling

    def forward(self, x):
        if self.scale:
            x = x * self.scale_param  # Shape: [batch_size, input_size]

        x = x.unsqueeze(2)  # Shape: [batch_size, input_size, 1]
        B = self.B.unsqueeze(0)  # Shape: [1, input_size, num_features_per_input]
        x_proj = 2 * np.pi * x * B  # Shape: [batch_size, input_size, num_features_per_input]

        # Compute sin and cos
        x_sin = torch.sin(x_proj)
        x_cos = torch.cos(x_proj)

        # Concatenate and flatten
        x_fourier = torch.cat([x_sin, x_cos], dim=2)  # Shape: [batch_size, input_size, num_features_per_input * 2]
        x_fourier = x_fourier.view(x.size(0), -1)  # Shape: [batch_size, total_fourier_features]
        return x_fourier

import numpy as np
import torch
from torch import nn


class FourierEncoder(nn.Module):
    def __init__(
            self,
            input_size,
            num_features_per_input,
            kernel_size=1,
            scale=True,
            init_frequency_range=(0, 5),
            use_feature_scaling=True
    ):
        """
        Optimized Learnable Fourier Transform Layer with a convolutional kernel for localized feature extraction.

        :param input_size: Number of input features.
        :param num_features_per_input: Number of Fourier features per input feature.
        :param kernel_size: Size of the kernel for local feature extraction.
        :param scale: Whether to scale the input before applying the Fourier Transform.
        :param init_frequency_range: Range for the initialization of the frequency matrix B.
        :param use_feature_scaling: Whether to add learnable scaling for the Fourier features.
        """
        super(FourierEncoder, self).__init__()
        self.scale = scale
        self.input_size = input_size
        self.num_features_per_input = num_features_per_input
        self.use_feature_scaling = use_feature_scaling

        # Learnable kernel for localized feature extraction (1D convolution)
        self.kernel = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2,
                                bias=False)

        # Initialize frequency matrix B with learnable parameters
        self.B = nn.Parameter(torch.empty(input_size, num_features_per_input))
        nn.init.uniform_(self.B, *init_frequency_range)

        # Optional per-feature scaling of input
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(input_size))

        # Optional scaling of Fourier features after transformation
        if self.use_feature_scaling:
            self.feature_scaling = nn.Parameter(torch.ones(input_size, num_features_per_input * 2))

        self.output_dim = num_features_per_input * 2

    def forward(self, x):
        # Scale the input (if enabled)
        if self.scale:
            x = x * self.scale_param  # Shape: [batch_size, input_size]

        # Add a channel dimension and apply kernel (convolution)
        x = x.unsqueeze(1)  # Shape: [batch_size, 1, input_size]
        x = self.kernel(x)  # Shape: [batch_size, 1, input_size] (after convolution)
        x = x.squeeze(1)  # Remove the channel dimension: [batch_size, input_size]

        # Unsqueeze the input for per-feature Fourier transformations
        x = x.unsqueeze(2)  # Shape: [batch_size, input_size, 1]
        B = self.B.unsqueeze(0)  # Shape: [1, input_size, num_features_per_input]

        # Project the input through frequencies
        x_proj = 2 * np.pi * x * B  # Shape: [batch_size, input_size, num_features_per_input]

        # Apply activation function (sin_cos, sin, cos, or tanh)
        x_sin = torch.sin(x_proj)
        x_cos = torch.cos(x_proj)
        x_fourier = torch.cat([x_sin, x_cos], dim=2)  # Concatenate sin and cos

        # Flatten the Fourier features
        x_fourier = x_fourier.view(x.size(0), -1)  # Shape: [batch_size, total_fourier_features]

        # Optionally scale the Fourier features
        if self.use_feature_scaling:
            x_fourier = x_fourier * self.feature_scaling.view(1, -1)

        return x_fourier

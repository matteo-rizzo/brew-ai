import torch
from torch import nn


class AdaptiveHermiteConvLayer(nn.Module):
    def __init__(self, input_size: int, max_degree: int, kernel_size: int = 3, stride: int = 1, normalize: bool = True):
        """
        Adaptive Hermite Layer with learnable scaling, bias, adaptive degree weighting, and a convolutional kernel.

        :param input_size: The size of the input features.
        :param max_degree: Maximum degree of the Hermite polynomials to compute.
        :param kernel_size: Size of the convolutional kernel applied to the Hermite polynomials.
        :param stride: Stride of the convolution.
        :param normalize: Whether to normalize the Hermite polynomial outputs.
        """
        super(AdaptiveHermiteConvLayer, self).__init__()
        assert max_degree >= 1, "Maximum degree of Hermite polynomials must be at least 1."
        self.input_size = input_size
        self.max_degree = max_degree
        self.normalize = normalize

        # Learnable scaling and bias for each degree
        self.scaling_factors = nn.Parameter(
            torch.ones(input_size, max_degree + 1))  # Shape: [input_size, max_degree + 1]
        self.bias = nn.Parameter(torch.zeros(input_size, max_degree + 1))  # Shape: [input_size, max_degree + 1]

        # Learnable combination weights for adaptive degree contribution
        self.degree_weights = nn.Parameter(
            torch.ones(input_size, max_degree + 1))  # Shape: [input_size, max_degree + 1]

        # Convolutional kernel applied over Hermite polynomial terms
        self.conv1d = nn.Conv1d(in_channels=max_degree + 1, out_channels=max_degree + 1, kernel_size=kernel_size,
                                stride=stride, padding=kernel_size // 2)

        # Optional normalization layer
        self.normalization_layer = nn.LayerNorm(input_size * (max_degree + 1)) if normalize else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Hermite polynomials with adaptive degree contributions and a convolutional kernel.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Transformed tensor with learnable Hermite polynomial features and convolution applied.
        """
        batch_size, input_size = x.size()
        assert input_size == self.input_size, f"Input size mismatch: expected {self.input_size}, got {input_size}"

        # Start with the zeroth and first Hermite polynomials: H_0(x) = 1, H_1(x) = 2x
        hermite_polys = [torch.ones_like(x), 2 * x]

        # Recurrence relation for Hermite polynomials: H_{n+1}(x) = 2xH_n(x) - 2nH_{n-1}(x)
        for n in range(1, self.max_degree):
            next_poly = 2 * x * hermite_polys[-1] - 2 * n * hermite_polys[-2]
            hermite_polys.append(next_poly)

        # Stack the polynomials along the degree axis
        hermite_polys = torch.stack(hermite_polys, dim=2)  # Shape: [batch_size, input_size, max_degree + 1]

        # Ensure scaling factors and bias are broadcasted to match the shape of hermite_polys
        scaling_factors = self.scaling_factors.unsqueeze(0)  # Shape: [1, input_size, max_degree + 1]
        bias = self.bias.unsqueeze(0)  # Shape: [1, input_size, max_degree + 1]

        # Apply learnable scaling, bias, and adaptive degree weighting for each polynomial degree
        hermite_polys = hermite_polys * scaling_factors + bias
        hermite_polys = hermite_polys * self.degree_weights.unsqueeze(
            0)  # Shape: [batch_size, input_size, max_degree + 1]

        # Permute for convolution: move degree axis to channel axis for Conv1d
        hermite_polys = hermite_polys.permute(0, 2, 1)  # Shape: [batch_size, max_degree + 1, input_size]

        # Apply 1D convolution over the Hermite polynomials
        hermite_polys_conv = self.conv1d(hermite_polys)  # Shape: [batch_size, max_degree + 1, input_size]

        # Permute back to [batch_size, input_size, degree]
        hermite_polys_conv = hermite_polys_conv.permute(0, 2, 1)

        # Flatten the feature dimensions
        hermite_polys_flat = hermite_polys_conv.reshape(batch_size, -1)  # Shape: [batch_size, input_size * new_degree]

        # Apply optional normalization
        if self.normalize:
            hermite_polys_flat = self.normalization_layer(hermite_polys_flat)

        return hermite_polys_flat

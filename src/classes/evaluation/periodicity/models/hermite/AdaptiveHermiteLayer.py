import torch
from torch import nn


class AdaptiveHermiteLayer(nn.Module):
    def __init__(self, input_size: int, max_degree: int, normalize: bool = True):
        """
        HermiteLayer with additional learnable scaling, bias, and adaptive degree weights for each polynomial term.

        :param input_size: The size of the input features.
        :param max_degree: Maximum degree of the Hermite polynomials to compute.
        :param normalize: Whether to normalize the Hermite polynomial outputs.
        """
        super(AdaptiveHermiteLayer, self).__init__()
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

        # Optional normalization layer
        self.normalization_layer = nn.LayerNorm(input_size * (max_degree + 1)) if normalize else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Hermite polynomials with adaptive degree contributions.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Transformed tensor with learnable Hermite polynomial features.
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

        # Apply softmax to the degree weights to get adaptive degree contributions
        adaptive_weights = torch.softmax(self.degree_weights, dim=-1).unsqueeze(
            0)  # Shape: [1, input_size, max_degree + 1]

        # Ensure scaling factors and bias are broadcasted to match the shape of hermite_polys
        scaling_factors = self.scaling_factors.unsqueeze(0)  # Shape: [1, input_size, max_degree + 1]
        bias = self.bias.unsqueeze(0)  # Shape: [1, input_size, max_degree + 1]

        # Apply learnable scaling, bias, and adaptive degree weighting for each polynomial degree
        hermite_polys = hermite_polys * scaling_factors + bias
        hermite_polys = hermite_polys * adaptive_weights  # Shape: [batch_size, input_size, max_degree + 1]

        # Flatten the feature dimension
        hermite_polys_flat = hermite_polys.view(batch_size, -1)  # Shape: [batch_size, input_size * (max_degree + 1)]

        # Apply optional normalization
        if self.normalize:
            hermite_polys_flat = self.normalization_layer(hermite_polys_flat)

        return hermite_polys_flat

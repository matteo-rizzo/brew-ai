import torch
from torch import nn


class AdaptiveChebyshevConvLayer(nn.Module):
    def __init__(self, input_size, max_terms, kernel_size=1, stride=1, scale=True, normalize=True):
        """
        Adaptive Chebyshev Polynomial Layer with convolutional kernel and learnable polynomial orders.

        :param input_size: Number of input features.
        :param max_terms: Maximum number of Chebyshev polynomial terms.
        :param kernel_size: Size of the convolutional kernel applied to the Chebyshev polynomials.
        :param stride: Stride of the convolution.
        :param scale: Whether to scale the input for stability.
        :param normalize: Whether to normalize the Chebyshev terms.
        """
        super(AdaptiveChebyshevConvLayer, self).__init__()
        self.max_terms = max_terms
        self.scale = scale
        self.normalize = normalize

        # Scaling parameters for input
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(input_size))

        # Padding needs to be adjusted based on kernel size
        padding = (kernel_size - 1) // 2  # Ensure the output size remains the same after convolution

        # Convolutional kernel applied over Chebyshev terms
        self.conv1d = nn.Conv1d(max_terms, max_terms, kernel_size=kernel_size, stride=stride, padding=padding)

        # Learnable polynomial importance weights (adaptive ordering)
        self.poly_weights = nn.Parameter(torch.ones(input_size, max_terms))

        # Normalization layer
        if self.normalize:
            self.norm_layer = nn.LayerNorm(input_size * max_terms)

    def forward(self, x):
        batch_size, input_size = x.shape

        # Scale the input
        if self.scale:
            x = x * self.scale_param

        # Generate Chebyshev polynomials up to the maximum order
        x = x.unsqueeze(2)  # Shape: [batch_size, input_size, 1]
        T = [torch.ones_like(x), x]  # T0 and T1

        for _ in range(2, self.max_terms):
            T_next = 2 * x * T[-1] - T[-2]
            T.append(T_next)

        # Stack the polynomials
        x_cheb = torch.cat(T, dim=2)  # Shape: [batch_size, input_size, max_terms]

        # Permute to match the Conv1d input format [batch_size, channels, length]
        x_cheb = x_cheb.permute(0, 2, 1)  # Shape: [batch_size, max_terms, input_size]

        # Apply convolutional kernel over Chebyshev terms
        x_cheb = self.conv1d(x_cheb)  # Shape: [batch_size, max_terms, input_size]

        # Permute back to [batch_size, input_size, max_terms] for further processing
        x_cheb = x_cheb.permute(0, 2, 1)  # Shape: [batch_size, input_size, max_terms]

        # Apply the learnable polynomial weights
        x_cheb_weighted = x_cheb * self.poly_weights.unsqueeze(0)  # Shape: [batch_size, input_size, max_terms]

        # Flatten the feature dimensions
        x_cheb_flat = x_cheb_weighted.reshape(batch_size, -1)  # Shape: [batch_size, input_size * max_terms]

        # Normalize the output
        if self.normalize:
            x_cheb_flat = self.norm_layer(x_cheb_flat)

        return x_cheb_flat

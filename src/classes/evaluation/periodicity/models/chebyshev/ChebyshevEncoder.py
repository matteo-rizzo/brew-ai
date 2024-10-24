import torch
from torch import nn

class ChebyshevEncoder(nn.Module):
    def __init__(self, input_size, max_terms, kernel_size=3, scale=True, normalize=True):
        """
        Adaptive Chebyshev Polynomial Layer with learnable scaling, polynomial importance weights, and kernel interactions.

        :param input_size: Number of input features.
        :param max_terms: Maximum number of Chebyshev polynomial terms.
        :param kernel_size: Size of the kernel applied to the Chebyshev polynomials.
        :param scale: Whether to apply learnable scaling to the input.
        :param normalize: Whether to apply normalization to the final output.
        """
        super(ChebyshevEncoder, self).__init__()
        self.max_terms = max_terms
        self.kernel_size = kernel_size
        self.scale = scale
        self.normalize = normalize

        # Optional scaling for each input feature
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(input_size))

        # Learnable weights for polynomial terms
        self.poly_weights = nn.Parameter(torch.randn(input_size, max_terms))

        # Learnable kernel for polynomial interactions
        self.kernel = nn.Parameter(torch.randn(input_size, max_terms, kernel_size))

        # Optional normalization layer
        if self.normalize:
            self.norm_layer = nn.LayerNorm(input_size * max_terms)

    def forward(self, x):
        """
        Forward pass of the ChebyshevEncoder.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Encoded tensor of shape [batch_size, input_size * max_terms].
        """
        batch_size, input_size = x.shape

        # Apply learnable scaling if enabled
        if self.scale:
            x = x * self.scale_param

        # Generate Chebyshev polynomials up to the maximum order
        x = x.unsqueeze(2)  # Shape: [batch_size, input_size, 1] to align for Chebyshev polynomials
        T = [torch.ones_like(x), x]  # T0 and T1 (Chebyshev base polynomials)

        for n in range(2, self.max_terms):
            T_next = 2 * x * T[-1] - T[-2]  # Recurrence relation for Chebyshev polynomials
            T.append(T_next)

        # Stack the polynomials along the last dimension
        x_cheb = torch.cat(T, dim=2)  # Shape: [batch_size, input_size, max_terms]

        # Apply the learnable kernel to modulate polynomial interactions
        x_cheb = torch.einsum('bim,imk->bik', x_cheb, self.kernel)  # Shape: [batch_size, input_size, kernel_size]

        # Apply the learnable weights for each polynomial order
        x_cheb_weighted = x_cheb * self.poly_weights.unsqueeze(0)  # Shape: [batch_size, input_size, max_terms]

        # Flatten the features to [batch_size, input_size * max_terms]
        x_cheb_flat = x_cheb_weighted.reshape(batch_size, -1)

        # Apply normalization if enabled
        if self.normalize:
            x_cheb_flat = self.norm_layer(x_cheb_flat)

        return x_cheb_flat

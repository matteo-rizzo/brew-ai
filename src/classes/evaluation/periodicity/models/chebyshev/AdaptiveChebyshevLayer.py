import torch
from torch import nn

class AdaptiveChebyshevLayer(nn.Module):
    def __init__(self, input_size, max_terms, kernel_size=3, scale=True, normalize=True):
        """
        Adaptive Chebyshev Polynomial Layer with kernel and learnable polynomial orders.

        :param input_size: Number of input features.
        :param max_terms: Maximum number of Chebyshev polynomial terms.
        :param kernel_size: Size of the kernel applied to the Chebyshev polynomials.
        :param scale: Whether to scale the input for stability.
        :param normalize: Whether to normalize the Chebyshev terms.
        """
        super(AdaptiveChebyshevLayer, self).__init__()
        self.max_terms = max_terms
        self.kernel_size = kernel_size
        self.scale = scale
        self.normalize = normalize

        # Learnable per-feature scaling
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(input_size))

        # Learnable polynomial importance weights (adaptive ordering)
        self.poly_weights = nn.Parameter(torch.ones(input_size, max_terms))

        # Kernel to modulate Chebyshev polynomial interactions
        self.kernel = nn.Parameter(torch.randn(input_size, max_terms, kernel_size))

        # Normalization layer
        if self.normalize:
            self.norm_layer = nn.LayerNorm(input_size * max_terms)

    def forward(self, x):
        batch_size, input_size = x.shape

        # Scale the input if scaling is enabled
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

        # Apply the learnable kernel
        x_cheb = torch.einsum('bim,imk->bik', x_cheb, self.kernel)  # Shape: [batch_size, input_size, kernel_size]

        # Apply the learnable polynomial weights to adaptively select orders
        x_cheb_weighted = x_cheb * self.poly_weights.unsqueeze(0)  # Shape: [batch_size, input_size, max_terms]

        # Flatten the feature dimensions
        x_cheb_flat = x_cheb_weighted.reshape(batch_size, -1)  # Shape: [batch_size, input_size * max_terms]

        # Normalize if enabled
        if self.normalize:
            x_cheb_flat = self.norm_layer(x_cheb_flat)

        return x_cheb_flat

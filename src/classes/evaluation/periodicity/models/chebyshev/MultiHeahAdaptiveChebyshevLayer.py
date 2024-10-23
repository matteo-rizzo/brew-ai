import torch
from torch import nn


class MultiHeadAdaptiveChebyshevLayer(nn.Module):
    def __init__(self, input_size, max_terms, num_heads=5, kernel_size=5, scale=True, normalize=True):
        """
        Multi-Head Adaptive Chebyshev Polynomial Layer with learnable polynomial orders and kernel.

        :param input_size: Number of input features.
        :param max_terms: Maximum number of Chebyshev polynomial terms per head.
        :param num_heads: Number of independent heads for Chebyshev polynomials.
        :param kernel_size: Size of the kernel applied to the Chebyshev polynomials.
        :param scale: Whether to scale the input for stability.
        :param normalize: Whether to normalize the Chebyshev terms.
        """
        super(MultiHeadAdaptiveChebyshevLayer, self).__init__()
        self.max_terms = max_terms
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.scale = scale
        self.normalize = normalize

        # Scaling for input features (per head)
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(num_heads, input_size))

        # Learnable polynomial importance weights for each head
        self.poly_weights = nn.Parameter(torch.ones(num_heads, input_size, max_terms))

        # Kernel to modulate Chebyshev polynomial interactions (per head)
        self.kernel = nn.Parameter(torch.randn(num_heads, input_size, max_terms, kernel_size))

        # Normalization layer (optional)
        if self.normalize:
            self.norm_layer = nn.LayerNorm(num_heads * input_size * kernel_size)

    def forward(self, x):
        batch_size, input_size = x.shape

        # Initialize a list to store the output of each head
        head_outputs = []

        for head in range(self.num_heads):
            # Scale the input for each head if scaling is enabled
            x_head = x * self.scale_param[head] if self.scale else x

            # Compute Chebyshev polynomials up to max_terms for this head
            x_head = x_head.unsqueeze(2)  # Shape: [batch_size, input_size, 1]
            T = [torch.ones_like(x_head), x_head]  # T0 and T1

            for _ in range(2, self.max_terms):
                T_next = 2 * x_head * T[-1] - T[-2]
                T.append(T_next)

            # Stack the polynomials
            x_cheb = torch.cat(T, dim=2)  # Shape: [batch_size, input_size, max_terms]

            # Apply the learnable kernel for this head
            x_cheb = torch.einsum('bim,imk->bik', x_cheb, self.kernel[head])  # Shape: [batch_size, input_size, kernel_size]

            # Apply the learnable polynomial weights for this head
            x_cheb_weighted = x_cheb * self.poly_weights[head].unsqueeze(0)  # Shape: [batch_size, input_size, kernel_size]

            # Flatten the feature dimensions
            x_cheb_flat = x_cheb_weighted.reshape(batch_size, -1)  # Shape: [batch_size, input_size * kernel_size]

            head_outputs.append(x_cheb_flat)

        # Concatenate the outputs of all heads
        x_multihead = torch.cat(head_outputs, dim=1)  # Shape: [batch_size, num_heads * input_size * kernel_size]

        # Apply normalization if enabled
        if self.normalize:
            x_multihead = self.norm_layer(x_multihead)

        return x_multihead

import torch
from torch import nn


class ChebyshevLayer(nn.Module):
    def __init__(self, input_size, num_terms, scale=True, normalize=True):
        """
        Optimized Chebyshev Polynomial Layer with vectorized computations.

        :param input_size: Number of input features.
        :param num_terms: Number of Chebyshev polynomial terms.
        :param scale: Whether to scale the input for stability.
        :param normalize: Whether to normalize the Chebyshev terms.
        """
        super(ChebyshevLayer, self).__init__()
        self.num_terms = num_terms
        self.scale = scale
        self.normalize = normalize

        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(input_size))  # Per-feature scaling

        if self.normalize:
            self.norm_layer = nn.LayerNorm(input_size * num_terms)

    def forward(self, x):
        if self.scale:
            x = x * self.scale_param

        x = x.unsqueeze(2)  # Shape: [batch_size, input_size, 1]
        T = [torch.ones_like(x), x]  # T0 and T1

        for _ in range(2, self.num_terms):
            T_next = 2 * x * T[-1] - T[-2]
            T.append(T_next)

        x_cheb = torch.cat(T, dim=2)  # Shape: [batch_size, input_size, num_terms]
        x_cheb = x_cheb.view(x.size(0), -1)  # Flatten

        if self.normalize:
            x_cheb = self.norm_layer(x_cheb)

        return x_cheb

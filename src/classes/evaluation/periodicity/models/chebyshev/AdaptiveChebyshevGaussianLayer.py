import torch
from torch import nn


class AdaptiveChebyshevGaussianLayer(nn.Module):
    def __init__(self, input_size, max_terms, kernel_std=1.0, scale=True, normalize=True):
        """
        Adaptive Chebyshev Polynomial Layer with a Gaussian kernel.

        :param input_size: Number of input features.
        :param max_terms: Maximum number of Chebyshev polynomial terms.
        :param kernel_std: Standard deviation for the Gaussian kernel.
        :param scale: Whether to scale the input for stability.
        :param normalize: Whether to normalize the Chebyshev terms.
        """
        super(AdaptiveChebyshevGaussianLayer, self).__init__()
        self.max_terms = max_terms
        self.kernel_std = kernel_std
        self.scale = scale
        self.normalize = normalize

        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(input_size))

        # Learnable polynomial importance weights (adaptive ordering)
        self.poly_weights = nn.Parameter(torch.ones(input_size, max_terms))

        # Gaussian kernel for smoothing the terms
        self.kernel = nn.Parameter(self._create_gaussian_kernel(input_size, max_terms, kernel_std))

        if self.normalize:
            self.norm_layer = nn.LayerNorm(input_size * max_terms)

    @staticmethod
    def _create_gaussian_kernel(input_size, max_terms, std):
        """
        Create a Gaussian kernel matrix for smoothing.

        :param input_size: Number of input features.
        :param max_terms: Number of Chebyshev terms.
        :param std: Standard deviation for the Gaussian kernel.
        :return: Gaussian kernel tensor.
        """
        kernel = torch.linspace(-1, 1, steps=max_terms).unsqueeze(0).expand(input_size, -1)
        kernel = torch.exp(-0.5 * (kernel ** 2) / (std ** 2))
        return kernel

    def forward(self, x):
        batch_size, input_size = x.shape

        if self.scale:
            x = x * self.scale_param

        x = x.unsqueeze(2)  # Shape: [batch_size, input_size, 1]
        T = [torch.ones_like(x), x]

        for _ in range(2, self.max_terms):
            T_next = 2 * x * T[-1] - T[-2]
            T.append(T_next)

        x_cheb = torch.cat(T, dim=2)  # Shape: [batch_size, input_size, max_terms]

        # Apply Gaussian kernel for smoothing
        x_cheb = x_cheb * self.kernel.unsqueeze(0)

        # Apply the learnable polynomial weights
        x_cheb_weighted = x_cheb * self.poly_weights.unsqueeze(0)

        x_cheb_flat = x_cheb_weighted.view(batch_size, -1)

        if self.normalize:
            x_cheb_flat = self.norm_layer(x_cheb_flat)

        return x_cheb_flat

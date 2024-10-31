import torch
from torch import nn


class ChebyshevEncoder(nn.Module):
    def __init__(
            self,
            input_size,
            max_terms,
            num_heads=4,
            kernel_size=5,
            scale=True,
            normalize=True,
            residual=False,
            activation=nn.SiLU()
    ):
        """
        Multi-Headed Chebyshev Polynomial Encoder with residual connections, dynamic kernel interactions, and optional non-linear activation.

        :param input_size: Number of input features.
        :param max_terms: Maximum number of Chebyshev polynomial terms.
        :param num_heads: Number of heads for multi-headed Chebyshev encoding.
        :param kernel_size: Size of the kernel for interaction modulation.
        :param scale: Whether to apply learnable scaling to each input.
        :param normalize: Whether to normalize the final output.
        :param residual: Whether to add residual connections.
        :param activation: Activation function after polynomial transformations.
        """
        super(ChebyshevEncoder, self).__init__()
        self.input_size = input_size
        self.max_terms = max_terms
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.scale = scale
        self.normalize = normalize
        self.residual = residual
        self.activation = activation

        # Optional scaling parameter for each input feature
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(input_size))

        # Separate learnable weights for each head
        self.poly_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_size, max_terms)) for _ in range(num_heads)
        ])

        # Separate learnable kernels for each head
        self.kernels = nn.ParameterList([
            nn.Parameter(torch.randn(input_size, max_terms, kernel_size)) for _ in range(num_heads)
        ])

        # Optional normalization layer
        if self.normalize:
            self.norm_layer = nn.LayerNorm(input_size * max_terms * num_heads)

        self.output_dim = max_terms * num_heads

    def forward(self, x):
        """
        Forward pass of the Multi-Headed ChebyshevEncoder.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Encoded tensor of shape [batch_size, input_size * max_terms * num_heads].
        """
        batch_size, input_size = x.shape

        # Apply scaling if enabled
        if self.scale:
            x = x * self.scale_param

        # Generate Chebyshev polynomials up to max_terms efficiently
        x = x.unsqueeze(2)  # Shape: [batch_size, input_size, 1]
        T = [torch.ones_like(x), x]  # T0 and T1

        # Recurrence for Chebyshev polynomials
        for n in range(2, self.max_terms):
            T_next = 2 * x * T[-1] - T[-2]
            T.append(T_next)

        # Stack polynomials along the last dimension
        x_cheb = torch.cat(T, dim=2)  # Shape: [batch_size, input_size, max_terms]

        # Multi-head Chebyshev encoding
        head_outputs = []
        for i in range(self.num_heads):
            # Apply the learnable kernel for this head
            x_head = torch.einsum('bim,imk->bik', x_cheb,
                                  self.kernels[i])  # Shape: [batch_size, input_size, kernel_size]

            # Apply learnable weights for this head
            x_head_weighted = x_head * self.poly_weights[i].unsqueeze(0)  # Shape: [batch_size, input_size, max_terms]

            # Apply residual connections if enabled
            if self.residual:
                x_residual = x.expand_as(x_head_weighted)
                x_head_weighted = x_head_weighted + x_residual

            # Apply activation function
            if self.activation:
                x_head_weighted = self.activation(x_head_weighted)

            # Flatten for this head and collect
            x_head_flat = x_head_weighted.reshape(batch_size, -1)
            head_outputs.append(x_head_flat)

        # Concatenate all heads
        x_multi_head = torch.cat(head_outputs, dim=1)  # Shape: [batch_size, input_size * max_terms * num_heads]

        # Normalize if enabled
        if self.normalize:
            x_multi_head = self.norm_layer(x_multi_head)

        return x_multi_head

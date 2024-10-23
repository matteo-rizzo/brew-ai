import torch
from torch import nn

from src.classes.evaluation.periodicity.feature_selector.FeatureSelector import FeatureSelector


class AttentionFeatureSelector(FeatureSelector):
    def __init__(self, input_size: int, embed_size: int = 16, num_heads: int = 4, dropout: float = 0.2):
        """
        Simplified Attention-based Feature Selector with Batch Normalization.

        :param input_size: Number of input features.
        :param embed_size: Embedding size for attention computations.
        :param num_heads: Number of attention heads.
        :param dropout: Dropout probability.
        """
        super(AttentionFeatureSelector, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size

        # Feature Embedding Layer
        self.embedding = nn.Linear(1, embed_size)

        # Batch Normalization Layer after Embedding
        self.bn1 = nn.BatchNorm1d(embed_size)

        # Multi-head Attention Layer
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=num_heads, batch_first=True
        )

        # Batch Normalization Layer after Attention
        self.bn2 = nn.BatchNorm1d(embed_size)

        # Output Layer
        self.output_layer = nn.Sequential(
            nn.Linear(embed_size, 1),
            nn.Sigmoid()
        )

        # Dropout Layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to estimate the periodicity of each input feature.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Periodicity scores of shape [batch_size, input_size].
        """
        # Reshape x to [batch_size, input_size, 1]
        x = x.unsqueeze(-1)  # Shape: [batch_size, input_size, 1]

        # Embed features
        x_embedded = self.embedding(x)  # Shape: [batch_size, input_size, embed_size]

        # Apply batch normalization over the embed_size dimension
        x_embedded = x_embedded.transpose(1, 2)  # Shape: [batch_size, embed_size, input_size]
        x_embedded = self.bn1(x_embedded)  # Normalize over batch and sequence dimensions
        x_embedded = x_embedded.transpose(1, 2)  # Shape: [batch_size, input_size, embed_size]

        # Apply dropout
        x_embedded = self.dropout(x_embedded)

        # Apply multi-head attention (self-attention)
        attn_output, _ = self.attention(
            x_embedded, x_embedded, x_embedded
        )  # Shape: [batch_size, input_size, embed_size]

        # Apply batch normalization after attention
        attn_output = attn_output.transpose(1, 2)  # Shape: [batch_size, embed_size, input_size]
        attn_output = self.bn2(attn_output)
        attn_output = attn_output.transpose(1, 2)  # Shape: [batch_size, input_size, embed_size]

        # Apply output layer to each feature
        out = self.output_layer(attn_output).squeeze(-1)  # Shape: [batch_size, input_size]

        return out

import torch
from torch import nn

from src.classes.evaluation.periodicity.feature_selector.FeatureSelector import FeatureSelector


class AttentionPeriodicityFeatureSelector(FeatureSelector):
    def __init__(self, input_size: int, embed_size: int = 8, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.2):
        """
        Enhanced Attention-based PeriodicityFeatureSelector using multi-layer self-attention over features.

        :param input_size: Number of input features.
        :param embed_size: Embedding size for attention computations.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of stacked attention layers.
        :param dropout: Dropout probability.
        """
        super(AttentionPeriodicityFeatureSelector, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size

        # Feature-wise Embedding
        self.embedding = nn.Sequential(nn.Linear(1, embed_size))

        # Batch Normalization after embedding
        self.batch_norm = nn.BatchNorm1d(embed_size)

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.SiLU(),
            nn.Linear(embed_size // 2, 1)
        )

        # Activation Function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, input_size]
        batch_size = x.size(0)

        # Reshape x to [batch_size * input_size, 1] for feature-wise processing
        x = x.view(-1, 1)  # Shape: [batch_size * input_size, 1]

        # Feature-wise Embedding
        x_embedded = self.embedding(x)  # Shape: [batch_size * input_size, embed_size]

        # Batch Normalization after embedding and reshape to [batch_size, input_size, embed_size]
        x_embedded = self.batch_norm(x_embedded.view(-1, self.embed_size))  # Apply batch norm on feature dimension
        x_embedded = x_embedded.view(batch_size, self.input_size, self.embed_size)

        # Transformer Encoder
        x_transformed = self.transformer_encoder(x_embedded)  # Shape: [batch_size, input_size, embed_size]

        # Output MLP applied to each feature
        out = self.output_mlp(x_transformed)  # Shape: [batch_size, input_size, 1]

        # Squeeze last dimension and apply sigmoid
        periodicity_scores = self.sigmoid(out.squeeze(-1))  # Shape: [batch_size, input_size]

        return periodicity_scores

from torch import nn


class AttentionPeriodicityFeatureSelector(nn.Module):
    def __init__(self, input_size, embed_size=64, num_heads=4):
        """
        Attention-based PeriodicityFeatureSelector using self-attention over features.

        :param input_size: Number of input features.
        :param embed_size: Embedding size for attention computations.
        :param num_heads: Number of attention heads.
        """
        super(AttentionPeriodicityFeatureSelector, self).__init__()
        self.embed_size = embed_size
        self.input_size = input_size
        self.num_heads = num_heads

        # Linear projection to embedding space
        self.embedding = nn.Linear(1, embed_size)

        # MultiheadAttention with batch_first=True to accept inputs as [batch_size, seq_length, embed_dim]
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)

        # Output layer to map from embed_size back to 1
        self.fc = nn.Linear(embed_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, input_size]
        # Reshape x to [batch_size, input_size, 1], treating features as sequence elements
        x = x.unsqueeze(-1)  # Shape: [batch_size, input_size, 1]

        # Embed x
        x_embedded = self.embedding(x)  # Shape: [batch_size, input_size, embed_size]

        # Compute self-attention over features
        # Since features are treated as sequence elements, attention is over the feature dimension
        attn_output, attn_weights = self.attention(x_embedded, x_embedded,
                                                   x_embedded)  # Outputs have shape [batch_size, input_size, embed_size]

        # Pass through output layer to get a single value per feature
        out = self.fc(attn_output)  # Shape: [batch_size, input_size, 1]

        # Squeeze last dimension and apply sigmoid to get periodicity scores between 0 and 1
        periodicity_scores = self.sigmoid(out.squeeze(-1))  # Shape: [batch_size, input_size]

        return periodicity_scores
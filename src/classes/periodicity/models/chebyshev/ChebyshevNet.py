from torch import nn

from src.classes.periodicity.models.chebyshev.ChebyshevLayer import ChebyshevLayer


class ChebyshevNet(nn.Module):
    def __init__(self, input_size, num_chebyshev_terms, hidden_size, num_layers=3, dropout_prob=0.2, batch_norm=True):
        """
        Improved ChebyshevNet with flexible number of layers, regularization, and learning rate scheduling.

        :param input_size: Size of the input features.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        :param hidden_size: Size of hidden layers.
        :param num_layers: Number of hidden layers (flexible).
        :param dropout_prob: Dropout probability for regularization.
        :param batch_norm: Whether to use batch normalization.
        """
        super(ChebyshevNet, self).__init__()
        self.chebyshev_layer = ChebyshevLayer(input_size, num_chebyshev_terms, normalize=True)

        total_features = input_size * num_chebyshev_terms  # Number of Chebyshev terms

        # Fully Connected Layers
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(total_features, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_prob))
            total_features = hidden_size  # Hidden layer size remains constant after the first layer

        # Output Layer
        layers.append(nn.Linear(hidden_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Apply Chebyshev transformation
        x_chebyshev = self.chebyshev_layer(x)
        # Feed into fully connected layers
        return self.network(x_chebyshev)

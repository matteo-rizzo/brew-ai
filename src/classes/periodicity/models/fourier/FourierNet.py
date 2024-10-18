from torch import nn

from src.classes.periodicity.models.fourier.FourierLayer import FourierLayer


class FourierNet(nn.Module):
    def __init__(self, input_size, num_fourier_features, hidden_size, num_layers=3, dropout_prob=0.2, batch_norm=True):
        """
        Improved FourierNet with flexible number of layers, regularization, and learning rate scheduling.

        :param input_size: Size of the input features.
        :param num_fourier_features: Number of Fourier features to learn.
        :param hidden_size: Size of hidden layers.
        :param num_layers: Number of hidden layers.
        :param dropout_prob: Dropout probability for regularization.
        :param batch_norm: Whether to use batch normalization.
        """
        super(FourierNet, self).__init__()
        self.fourier_layer = FourierLayer(input_size, num_fourier_features)

        total_features = input_size * num_fourier_features * 2  # We have sine and cosine outputs

        # Fully Connected Layers with flexible depth
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(total_features, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_prob))
            total_features = hidden_size  # After the first layer, total_features = hidden_size

        # Output Layer
        layers.append(nn.Linear(hidden_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Apply Fourier transformation to input features
        x_fourier = self.fourier_layer(x)
        # Feed Fourier-transformed features into fully connected layers
        return self.network(x_fourier)

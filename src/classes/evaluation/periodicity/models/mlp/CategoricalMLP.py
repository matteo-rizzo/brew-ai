from torch import nn


class CategoricalMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout_prob: float = 0.2):
        """
        MLP for handling one-hot encoded categorical features.

        :param input_size: Size of the input (one-hot encoded features).
        :param hidden_size: Size of hidden layers.
        :param num_layers: Number of hidden layers.
        :param dropout_prob: Dropout probability for regularization.
        """
        super(CategoricalMLP, self).__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            input_size = hidden_size

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the categorical MLP.

        :param x: Categorical input (one-hot encoded).
        :return: Processed categorical features.
        """
        return self.network(x)

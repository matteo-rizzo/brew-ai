from torch import nn, Tensor

from src.classes.evaluation.periodicity.models.chebyshev.ChebyshevLayer import ChebyshevLayer
from src.classes.evaluation.periodicity.models.mlp.MLPRegressor import MLPRegressor


class ChebyshevNet(nn.Module):
    def __init__(self, input_size: int, num_chebyshev_terms: int):
        """
        ChebyshevNet integrates Chebyshev transformations with an MLP regressor.

        :param input_size: Size of the input features.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        """
        super(ChebyshevNet, self).__init__()

        self.chebyshev_layer = ChebyshevLayer(input_size, num_chebyshev_terms, normalize=True)

        total_features = input_size * num_chebyshev_terms  # Total number of Chebyshev features

        # MLPRegressor for handling the fully connected layers
        self.mlp_regressor = MLPRegressor(total_features)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # Apply Chebyshev transformation
        x_chebyshev = self.chebyshev_layer(x)

        # Pass through the MLP regressor
        out = self.mlp_regressor(x_chebyshev)

        return out

from torch import Tensor
from torch import nn

from src.classes.evaluation.periodicity.models.chebyshev.ChebyshevBlock import ChebyshevBlock
from src.classes.evaluation.periodicity.models.classifier.MLPClassifier import MLPClassifier
from src.classes.evaluation.periodicity.models.regressor.MLPRegressor import MLPRegressor


class ChebyshevNet(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_chebyshev_terms: int,
            num_layers: int = 1,
            compression_dim: int = 128,
            dropout_prob: float = 0.2,
            output_size: int = 1
    ):
        """
        ChebyshevNet integrates Chebyshev transformations with an MLP regressor and adds a residual connection.

        :param input_size: Size of the input features.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        :param num_layers: Number of Chebyshev layers to stack.
        :param compression_dim: Dimension to compress features to between layers.
        :param dropout_prob: Probability of dropout after normalization layers (0 to 1).
        :param output_size: Output size; if > 1, a classifier is used; otherwise, a regressor.
        """
        super(ChebyshevNet, self).__init__()

        # Chebyshev Block
        self.chebyshev_block = ChebyshevBlock(
            input_size=input_size,
            num_chebyshev_terms=num_chebyshev_terms,
            num_layers=num_layers,
            compression_dim=compression_dim,
            dropout_prob=dropout_prob
        )

        total_features = self.chebyshev_block.output_dim

        # Choose between MLPClassifier and MLPRegressor based on output_size
        if output_size > 1:
            self.mlp = MLPClassifier(input_size=total_features, output_size=output_size)
        else:
            self.mlp = MLPRegressor(input_size=total_features)

    def forward(self, x: Tensor) -> Tensor:
        # Apply Chebyshev transformation through the ChebyshevBlock
        x_chebyshev = self.chebyshev_block(x)

        # Pass through the MLP regressor
        out = self.mlp(x_chebyshev)

        return out

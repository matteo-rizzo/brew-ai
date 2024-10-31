from src.classes.evaluation.periodicity.models.base.BaseBlock import BaseBlock
from src.classes.evaluation.periodicity.models.chebyshev.ChebyshevEncoder import ChebyshevEncoder


class ChebyshevBlock(BaseBlock):
    def __init__(
            self,
            input_size: int,
            num_chebyshev_terms: int,
            num_layers: int,
            compression_dim: int,
            dropout_prob: float
    ):
        """
        ChebyshevBlock applies a series of ChebyshevEncoder layers to transform the input features, with optional
        compression, batch normalization, and dropout between layers.

        :param input_size: Number of input features.
        :param num_chebyshev_terms: Number of terms in the Chebyshev polynomial for each layer.
        :param num_layers: Number of stacked ChebyshevEncoder layers.
        :param compression_dim: Size to compress features to between layers; if None, defaults to input size.
        :param dropout_prob: Dropout probability after each normalization layer.
        """
        super().__init__(
            input_size=input_size,
            encoder=ChebyshevEncoder,
            encoder_params={"max_terms": num_chebyshev_terms},
            num_layers=num_layers,
            compression_dim=compression_dim,
            dropout_prob=dropout_prob,
        )

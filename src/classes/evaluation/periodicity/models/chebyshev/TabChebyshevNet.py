from src.classes.evaluation.periodicity.models.base.BaseTabNet import BaseTabNet
from src.classes.evaluation.periodicity.models.chebyshev.ChebyshevBlock import ChebyshevBlock


class TabChebyshevNet(BaseTabNet):
    def __init__(
            self,
            continuous_input_size: int,
            categorical_input_size: int,
            num_chebyshev_terms: int,
            hidden_size: int,
            output_size: int,
            num_layers: int = 1,
            compression_dim: int = 1,
            dropout_prob: float = 0.2,
            use_residual: bool = True
    ):
        """
        TabChebyshevNet integrates Chebyshev transformations for continuous features with an MLP for categorical
        features, optionally including a residual connection for continuous features.

        :param continuous_input_size: Number of continuous input features.
        :param categorical_input_size: Number of one-hot encoded categorical features.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms to use in ChebyshevBlock.
        :param hidden_size: Size of hidden layers for categorical feature processing.
        :param output_size: Size of the network output; >1 indicates classification, 1 indicates regression.
        :param use_residual: Whether to apply a residual connection to continuous features.
        """
        # Initialize the ChebyshevBlock for processing continuous features
        chebyshev_layer = ChebyshevBlock(
            input_size=continuous_input_size,
            num_chebyshev_terms=num_chebyshev_terms,
            num_layers=num_layers,
            compression_dim=compression_dim,
            dropout_prob=dropout_prob
        )

        # Initialize the BaseTabNet with ChebyshevBlock for continuous features and CategoricalMLP for categorical
        super(TabChebyshevNet, self).__init__(
            processing_layer=chebyshev_layer,
            categorical_input_size=categorical_input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            use_residual=use_residual
        )

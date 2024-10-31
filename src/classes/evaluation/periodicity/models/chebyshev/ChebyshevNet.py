from src.classes.evaluation.periodicity.models.base.BaseNet import BaseNet
from src.classes.evaluation.periodicity.models.chebyshev.ChebyshevBlock import ChebyshevBlock


class ChebyshevNet(BaseNet):
    def __init__(
            self,
            input_size: int,
            num_chebyshev_terms: int,
            num_layers: int = 1,
            compression_dim: int = None,
            dropout_prob: float = 0.2,
            output_size: int = 1,
            use_residual: bool = True
    ):
        """
        ChebyshevNet combines a Chebyshev transformation block with an MLP-based network. Optionally includes
        a residual connection for enhanced stability.

        :param input_size: Size of the input features.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        :param num_layers: Number of Chebyshev layers to stack.
        :param compression_dim: Dimension to compress features to between layers.
        :param dropout_prob: Probability of dropout after normalization layers (0 to 1).
        :param output_size: Size of the network output; >1 indicates classification, 1 indicates regression.
        :param use_residual: Whether to include a residual connection between input and output.
        """
        # Initialize the Chebyshev transformation block
        chebyshev_block = ChebyshevBlock(
            input_size=input_size,
            num_chebyshev_terms=num_chebyshev_terms,
            num_layers=num_layers,
            compression_dim=compression_dim,
            dropout_prob=dropout_prob
        )

        # Initialize the BaseNet with the processing layer and other parameters
        super().__init__(processing_layer=chebyshev_block, output_size=output_size, use_residual=use_residual)

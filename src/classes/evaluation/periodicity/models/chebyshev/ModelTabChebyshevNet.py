from src.classes.evaluation.periodicity.models.base.BaseTabModel import BaseTabModel
from src.classes.evaluation.periodicity.models.chebyshev.TabChebyshevNet import TabChebyshevNet


class ModelTabChebyshevNet(BaseTabModel):
    def __init__(
            self,
            continuous_input_size: int,
            categorical_input_size: int,
            num_chebyshev_terms: int,
            hidden_size: int,
            output_size: int
    ):
        """
        ModelTabChebyshevNet initializes a TabChebyshevNet within the BaseTabModel framework, allowing for
        tabular data processing with both continuous and categorical inputs using Chebyshev transformations.

        :param continuous_input_size: Number of continuous input features.
        :param categorical_input_size: Number of categorical (one-hot encoded) input features.
        :param num_chebyshev_terms: Number of terms in the Chebyshev polynomial for feature transformation.
        :param hidden_size: Size of hidden layers for categorical feature processing.
        :param output_size: Size of the model's output; if >1, supports multi-output tasks.
        """
        # Initialize the TabChebyshevNet with specified parameters
        network = TabChebyshevNet(
            continuous_input_size=continuous_input_size,
            categorical_input_size=categorical_input_size,
            num_chebyshev_terms=num_chebyshev_terms,
            hidden_size=hidden_size,
            output_size=output_size
        )

        # Initialize the BaseTabModel with the configured network
        super(ModelTabChebyshevNet, self).__init__(network=network)

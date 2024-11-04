from src.classes.periodicity.models.autopnp.TabAutoPNPNet import TabAutoPNPNet
from src.classes.periodicity.models.base.BaseTabModel import BaseTabModel


class ModelTabAutoPNPNet(BaseTabModel):
    def __init__(
            self,
            continuous_input_size: int,
            categorical_input_size: int,
            num_fourier_features: int,
            max_poly_terms: int,
            hidden_size: int,
            output_size: int
    ):
        """
        ModelTabAutoPNPNet initializes a TabAutoPNPNet within the BaseTabModel framework, enabling feature
        transformations for both continuous and categorical inputs with Fourier and Chebyshev encodings.

        :param continuous_input_size: Number of continuous input features.
        :param categorical_input_size: Number of categorical (one-hot encoded) input features.
        :param num_fourier_features: Number of Fourier features generated per continuous input feature.
        :param max_poly_terms: Number of Chebyshev polynomial terms for continuous feature transformation.
        :param hidden_size: Size of hidden layers for processing categorical features.
        :param output_size: Desired output size; >1 indicates multi-output, 1 for single-output tasks.
        """
        # Initialize the TabAutoPNPNet with specified parameters
        network = TabAutoPNPNet(
            continuous_input_size=continuous_input_size,
            categorical_input_size=categorical_input_size,
            num_fourier_features=num_fourier_features,
            max_poly_terms=max_poly_terms,
            hidden_size=hidden_size,
            output_size=output_size
        )

        # Initialize BaseTabModel with the configured network
        super(ModelTabAutoPNPNet, self).__init__(network=network)

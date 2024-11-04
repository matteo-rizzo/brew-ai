from src.classes.periodicity.models.base.BaseTabModel import BaseTabModel
from src.classes.periodicity.models.fourier.TabFourierNet import TabFourierNet


class ModelTabFourierNet(BaseTabModel):
    def __init__(
            self,
            continuous_input_size: int,
            categorical_input_size: int,
            num_fourier_features: int,
            hidden_size: int,
            output_size: int
    ):
        """
        ModelTabFourierNet initializes a TabFourierNet within the BaseTabModel framework, supporting tabular data
        processing with Fourier transformations for continuous features and an MLP for categorical features.

        :param continuous_input_size: Number of continuous input features.
        :param categorical_input_size: Number of categorical (one-hot encoded) input features.
        :param num_fourier_features: Number of Fourier features to generate for each continuous input feature.
        :param hidden_size: Size of hidden layers for processing categorical features.
        :param output_size: Size of the model's output; if >1, supports multi-output tasks.
        """
        # Initialize the TabFourierNet with specified parameters
        network = TabFourierNet(
            continuous_input_size=continuous_input_size,
            categorical_input_size=categorical_input_size,
            num_fourier_features=num_fourier_features,
            hidden_size=hidden_size,
            output_size=output_size
        )

        # Initialize the BaseTabModel with the configured network
        super(ModelTabFourierNet, self).__init__(network=network)

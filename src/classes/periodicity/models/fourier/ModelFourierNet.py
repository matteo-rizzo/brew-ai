from src.classes.periodicity.models.base.BaseModel import BaseModel
from src.classes.periodicity.models.fourier.FourierNet import FourierNet


class ModelFourierNet(BaseModel):
    def __init__(self, input_size: int, num_fourier_features: int, output_size: int):
        """
        ModelFourierNet initializes a FourierNet within the BaseModel framework, allowing for
        data processing using Fourier transformations.

        :param input_size: Number of input features.
        :param num_fourier_features: Number of Fourier features to generate for each input feature.
        :param output_size: Size of the model's output; if >1, supports multi-output tasks.
        """
        # Initialize the FourierNet with specified parameters
        network = FourierNet(
            input_size=input_size,
            num_fourier_features=num_fourier_features,
            output_size=output_size
        )

        # Initialize the BaseModel with the configured network
        super(ModelFourierNet, self).__init__(network=network)

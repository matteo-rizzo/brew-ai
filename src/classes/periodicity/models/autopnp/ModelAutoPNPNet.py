from src.classes.periodicity.models.autopnp.AutoPNPNet import AutoPNPNet
from src.classes.periodicity.models.base.BaseModel import BaseModel


class ModelAutoPNPNet(BaseModel):
    def __init__(
            self,
            input_size: int,
            num_fourier_features: int,
            max_poly_terms: int,
            output_size: int
    ):
        """
        ModelAutoPNPNet initializes an AutoPNPNet within the BaseModel framework, using Fourier and Chebyshev
        transformations for feature learning.

        :param input_size: Number of input features.
        :param num_fourier_features: Number of Fourier features to generate per input feature.
        :param max_poly_terms: Number of Chebyshev polynomial terms for each layer.
        :param output_size: Desired output size; >1 indicates multi-output, 1 for single-output tasks.
        """
        # Initialize the AutoPNPNet with specified parameters
        network = AutoPNPNet(
            input_size=input_size,
            num_fourier_features=num_fourier_features,
            max_poly_terms=max_poly_terms,
            output_size=output_size
        )

        # Initialize BaseModel with the configured network
        super(ModelAutoPNPNet, self).__init__(network=network)

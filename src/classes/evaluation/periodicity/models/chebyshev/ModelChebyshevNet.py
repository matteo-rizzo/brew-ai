from src.classes.evaluation.periodicity.models.base.BaseModel import BaseModel
from src.classes.evaluation.periodicity.models.chebyshev.ChebyshevNet import ChebyshevNet


class ModelChebyshevNet(BaseModel):
    def __init__(self, input_size: int, num_chebyshev_terms: int, output_size: int):
        """
        ModelChebyshevNet is a wrapper for ChebyshevNet within the BaseModel framework, initializing it with specified
        input, output, and Chebyshev term settings.

        :param input_size: Number of input features for the model.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms in each layer.
        :param output_size: Desired size of the output.
        """
        # Initialize the ChebyshevNet with provided parameters
        network = ChebyshevNet(input_size=input_size, num_chebyshev_terms=num_chebyshev_terms, output_size=output_size)
        super().__init__(network=network)

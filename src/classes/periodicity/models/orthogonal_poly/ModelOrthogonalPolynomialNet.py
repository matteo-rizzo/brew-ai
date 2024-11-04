from src.classes.periodicity.models.base.BaseModel import BaseModel
from src.classes.periodicity.models.orthogonal_poly.OrthogonalPolynomialNet import OrthogonalPolynomialNet


class ModelOrthogonalPolynomialNet(BaseModel):
    def __init__(self, input_size: int, max_poly_terms: int, output_size: int):
        """
        ModelOrthogonalPolynomialNet is a wrapper for OrthogonalPolynomialNet within the BaseModel framework, initializing it with specified
        input, output, and Chebyshev term settings.

        :param input_size: Number of input features for the model.
        :param max_poly_terms: Number of Chebyshev polynomial terms in each layer.
        :param output_size: Desired size of the output.
        """
        # Initialize the OrthogonalPolynomialNet with provided parameters
        network = OrthogonalPolynomialNet(input_size=input_size, max_poly_terms=max_poly_terms, output_size=output_size)
        super().__init__(network=network)

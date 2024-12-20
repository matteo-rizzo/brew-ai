from typing import Union

from src.classes.periodicity.models.autopnp.ModelAutoPNPNet import ModelAutoPNPNet
from src.classes.periodicity.models.autopnp.ModelTabAutoPNPNet import ModelTabAutoPNPNet
from src.classes.periodicity.models.base.BaseModel import BaseModel
from src.classes.periodicity.models.base.BaseTabModel import BaseTabModel
from src.classes.periodicity.models.baseline.ModelBaseline import ModelBaseline
from src.classes.periodicity.models.fourier.ModelFourierNet import ModelFourierNet
from src.classes.periodicity.models.fourier.ModelTabFourierNet import ModelTabFourierNet
from src.classes.periodicity.models.orthogonal_poly.ModelOrthogonalPolynomialNet import ModelOrthogonalPolynomialNet
from src.classes.periodicity.models.orthogonal_poly.ModelTabOrthogonalPolynomialNet import \
    ModelTabOrthogonalPolynomialNet
from src.classes.periodicity.models.pnp.ModelPNPNet import ModelPNPNet
from src.classes.periodicity.models.pnp.ModelTabPNPNet import ModelTabPNPNet
from src.config import NUM_FOURIER_FEATURES, MAX_POLY_TERMS, CAT_HIDDEN_SIZE


class ModelFactory:

    def __init__(
            self,
            num_periodic_input_size: int = None,
            num_non_periodic_input_size: int = None,
            num_input_size: int = None,
            cat_input_size: int = None,
            output_size: int = 1,
            dataset_config: dict = None,
            num_fourier_features: int = NUM_FOURIER_FEATURES,
            max_poly_terms: int = MAX_POLY_TERMS,
            hidden_size: int = CAT_HIDDEN_SIZE,
    ):
        if not num_input_size:
            num_input_size = num_periodic_input_size + num_non_periodic_input_size

        self.model_constructors = {
            "fnet": lambda: ModelFourierNet(
                input_size=num_input_size,
                num_fourier_features=num_fourier_features,
                output_size=output_size
            ),
            "tabfnet": lambda: ModelTabFourierNet(
                continuous_input_size=num_input_size,
                categorical_input_size=cat_input_size,
                num_fourier_features=num_fourier_features,
                hidden_size=hidden_size,
                output_size=output_size
            ),
            "opnet": lambda: ModelOrthogonalPolynomialNet(
                input_size=num_input_size,
                max_poly_terms=max_poly_terms,
                output_size=output_size
            ),
            "tabcnet": lambda: ModelTabOrthogonalPolynomialNet(
                continuous_input_size=num_input_size,
                categorical_input_size=cat_input_size,
                max_poly_terms=max_poly_terms,
                hidden_size=hidden_size,
                output_size=output_size
            ),
            "pnpnet": lambda: ModelPNPNet(
                periodic_input_size=num_periodic_input_size,
                non_periodic_input_size=num_non_periodic_input_size,
                num_fourier_features=num_fourier_features,
                max_poly_terms=max_poly_terms,
                output_size=output_size
            ),
            "tabpnpnet": lambda: ModelTabPNPNet(
                periodic_input_size=num_periodic_input_size,
                non_periodic_input_size=num_non_periodic_input_size,
                categorical_input_size=cat_input_size,
                num_fourier_features=num_fourier_features,
                max_poly_terms=max_poly_terms,
                hidden_size=hidden_size,
                output_size=output_size
            ),
            "autopnpnet": lambda: ModelAutoPNPNet(
                input_size=num_input_size,
                num_fourier_features=num_fourier_features,
                max_poly_terms=max_poly_terms,
                output_size=output_size
            ),
            "tabautopnpnet": lambda: ModelTabAutoPNPNet(
                continuous_input_size=num_input_size,
                categorical_input_size=cat_input_size,
                num_fourier_features=num_fourier_features,
                max_poly_terms=max_poly_terms,
                hidden_size=hidden_size,
                output_size=output_size
            ),
            "tabbaseline": lambda: ModelBaseline(dataset_config, output_size)
        }

    def get_model(self, model_name: str) -> Union[BaseModel, BaseTabModel]:
        """
        Lazily initialize and return the requested model.

        :param model_name: Name of the model to initialize.
        :return: Initialized PyTorch model.
        :raises ValueError: If the model_name is not valid.
        """
        if model_name not in self.model_constructors:
            raise ValueError(f"Model '{model_name}' is not recognized. "
                             f"Available models: {list(self.model_constructors.keys())}")

        return self.model_constructors[model_name]()

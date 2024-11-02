from torch import nn

from src.classes.evaluation.periodicity.models.autopnp.ModelAutoPNPNet import ModelAutoPNPNet
from src.classes.evaluation.periodicity.models.autopnp.ModelTabAutoPNPNet import ModelTabAutoPNPNet
from src.classes.evaluation.periodicity.models.baseline.ModelBaseline import ModelBaseline
from src.classes.evaluation.periodicity.models.orthogonal_poly.ModelOrthogonalPolynomialNet import ModelOrthogonalPolynomialNet
from src.classes.evaluation.periodicity.models.orthogonal_poly.ModelTabOrthogonalPolynomialNet import ModelTabOrthogonalPolynomialNet
from src.classes.evaluation.periodicity.models.fourier.ModelFourierNet import ModelFourierNet
from src.classes.evaluation.periodicity.models.fourier.ModelTabFourierNet import ModelTabFourierNet
from src.classes.evaluation.periodicity.models.pnp.ModelPNPNet import ModelPNPNet
from src.classes.evaluation.periodicity.models.pnp.ModelTabPNPNet import ModelTabPNPNet
from src.config import NUM_FOURIER_FEATURES, MAX_POLY_TERMS, CAT_HIDDEN_SIZE


class ModelFactory:

    def __init__(
            self,
            num_periodic_input_size: int,
            num_non_periodic_input_size: int,
            cat_input_size: int,
            output_size: int,
            dataset_config: dict,
            num_fourier_features: int = NUM_FOURIER_FEATURES,
            max_poly_terms: int = MAX_POLY_TERMS,
            hidden_size: int = CAT_HIDDEN_SIZE,
    ):
        self.models = {
            "fnet": ModelFourierNet(
                input_size=num_periodic_input_size + num_non_periodic_input_size,
                num_fourier_features=num_fourier_features,
                output_size=output_size
            ),
            "tabfnet": ModelTabFourierNet(
                continuous_input_size=num_periodic_input_size + num_non_periodic_input_size,
                categorical_input_size=cat_input_size,
                num_fourier_features=num_fourier_features,
                hidden_size=hidden_size,
                output_size=output_size
            ),
            "opnet": ModelOrthogonalPolynomialNet(
                input_size=num_periodic_input_size + num_non_periodic_input_size,
                max_poly_terms=max_poly_terms,
                output_size=output_size
            ),
            "tabcnet": ModelTabOrthogonalPolynomialNet(
                continuous_input_size=num_periodic_input_size + num_non_periodic_input_size,
                categorical_input_size=cat_input_size,
                max_poly_terms=max_poly_terms,
                hidden_size=hidden_size,
                output_size=output_size
            ),
            "pnpnet": ModelPNPNet(
                periodic_input_size=num_periodic_input_size,
                non_periodic_input_size=num_non_periodic_input_size,
                num_fourier_features=num_fourier_features,
                max_poly_terms=max_poly_terms,
                output_size=output_size
            ),
            "tabpnpnet": ModelTabPNPNet(
                periodic_input_size=num_periodic_input_size,
                non_periodic_input_size=num_non_periodic_input_size,
                categorical_input_size=cat_input_size,
                num_fourier_features=num_fourier_features,
                max_poly_terms=max_poly_terms,
                hidden_size=hidden_size,
                output_size=output_size
            ),
            "autopnpnet": ModelAutoPNPNet(
                input_size=num_periodic_input_size + num_non_periodic_input_size,
                num_fourier_features=num_fourier_features,
                max_poly_terms=max_poly_terms,
                output_size=output_size
            ),
            "tabautopnpnet": ModelTabAutoPNPNet(
                continuous_input_size=num_periodic_input_size + num_non_periodic_input_size,
                categorical_input_size=cat_input_size,
                num_fourier_features=num_fourier_features,
                max_poly_terms=max_poly_terms,
                hidden_size=hidden_size,
                output_size=output_size
            ),
            "tabbaseline": ModelBaseline(dataset_config, output_size)
        }

    def get_model(self, model_name: str) -> nn.Module:
        return self.models[model_name]

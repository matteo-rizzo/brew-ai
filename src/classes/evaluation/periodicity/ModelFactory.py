from torch import nn

from src.classes.evaluation.periodicity.models.autopnp.ModelAutoPNPNet import ModelAutoPNPNet
from src.classes.evaluation.periodicity.models.autopnp.ModelTabAutoPNPNet import ModelTabAutoPNPNet
from src.classes.evaluation.periodicity.models.baseline.ModelBaseline import ModelBaseline
from src.classes.evaluation.periodicity.models.chebyshev.ModelChebyshevNet import ModelChebyshevNet
from src.classes.evaluation.periodicity.models.chebyshev.ModelTabChebyshevNet import ModelTabChebyshevNet
from src.classes.evaluation.periodicity.models.fourier.ModelFourierNet import ModelFourierNet
from src.classes.evaluation.periodicity.models.fourier.ModelTabFourierNet import ModelTabFourierNet
from src.classes.evaluation.periodicity.models.pnp.ModelPNPNet import ModelPNPNet
from src.classes.evaluation.periodicity.models.pnp.ModelTabPNPNet import ModelTabPNPNet
from src.config import NUM_FOURIER_FEATURES, NUM_CHEBYSHEV_TERMS, HIDDEN_SIZE


class ModelFactory:

    def __init__(
            self,
            num_periodic_input_size: int,
            num_non_periodic_input_size: int,
            cat_input_size: int,
            output_size: int,
            dataset_config: dict,
            num_fourier_features: int = NUM_FOURIER_FEATURES,
            num_chebyshev_terms: int = NUM_CHEBYSHEV_TERMS,
            hidden_size: int = HIDDEN_SIZE,
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
            "cnet": ModelChebyshevNet(
                input_size=num_periodic_input_size + num_non_periodic_input_size,
                num_chebyshev_terms=num_chebyshev_terms,
                output_size=output_size
            ),
            "tabcnet": ModelTabChebyshevNet(
                continuous_input_size=num_periodic_input_size + num_non_periodic_input_size,
                categorical_input_size=cat_input_size,
                num_chebyshev_terms=num_chebyshev_terms,
                hidden_size=hidden_size,
                output_size=output_size
            ),
            "pnpnet": ModelPNPNet(
                periodic_input_size=num_periodic_input_size,
                non_periodic_input_size=num_non_periodic_input_size,
                num_fourier_features=num_fourier_features,
                num_chebyshev_terms=num_chebyshev_terms,
                output_size=output_size
            ),
            "tabpnpnet": ModelTabPNPNet(
                periodic_input_size=num_periodic_input_size,
                non_periodic_input_size=num_non_periodic_input_size,
                categorical_input_size=cat_input_size,
                num_fourier_features=num_fourier_features,
                num_chebyshev_terms=num_chebyshev_terms,
                hidden_size=hidden_size,
                output_size=output_size
            ),
            "autopnpnet": ModelAutoPNPNet(
                input_size=num_periodic_input_size + num_non_periodic_input_size,
                num_fourier_features=num_fourier_features,
                num_chebyshev_terms=num_chebyshev_terms,
                output_size=output_size
            ),
            "tabautopnpnet": ModelTabAutoPNPNet(
                continuous_input_size=num_periodic_input_size + num_non_periodic_input_size,
                categorical_input_size=cat_input_size,
                num_fourier_features=num_fourier_features,
                num_chebyshev_terms=num_chebyshev_terms,
                hidden_size=hidden_size,
                output_size=output_size
            ),
            "tabbaseline": ModelBaseline(dataset_config, output_size)
        }

    def get_model(self, model_name: str) -> nn.Module:
        return self.models[model_name]

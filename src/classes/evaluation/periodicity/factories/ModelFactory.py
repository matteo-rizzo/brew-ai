from torch import nn

from src.classes.evaluation.periodicity.models.autopnp.ModelAutoPNPNet import ModelAutoPNPNet
from src.classes.evaluation.periodicity.models.autopnp.ModelTabAutoPNPNet import ModelTabAutoPNPNet
from src.classes.evaluation.periodicity.models.chebyshev.ModelChebyshevNet import ModelChebyshevNet
from src.classes.evaluation.periodicity.models.chebyshev.ModelTabChebyshevNet import ModelTabChebyshevNet
from src.classes.evaluation.periodicity.models.fourier.ModelFourierNet import ModelFourierNet
from src.classes.evaluation.periodicity.models.fourier.ModelTabFourierNet import ModelTabFourierNet
from src.classes.evaluation.periodicity.models.hermite.ModelHermiteNet import ModelHermiteNet
from src.classes.evaluation.periodicity.models.pnp.ModelPNPNet import ModelPNPNet
from src.classes.evaluation.periodicity.models.pnp.ModelTabPNPNet import ModelTabPNPNet


class ModelFactory:

    def __init__(
            self,
            num_periodic_input_size: int,
            num_non_periodic_input_size: int,
            cat_input_size: int,
            num_fourier_features: int = 15,
            num_chebyshev_terms: int = 3,
            hermite_degree: int = 3,
            hidden_size: int = 256
    ):
        self.models = {
            "hnet": ModelHermiteNet(
                input_size=num_periodic_input_size + num_non_periodic_input_size,
                hermite_degree=hermite_degree
            ),
            "fnet": ModelFourierNet(
                input_size=num_periodic_input_size + num_non_periodic_input_size,
                num_fourier_features=num_fourier_features
            ),
            "tabfnet": ModelTabFourierNet(
                continuous_input_size=num_periodic_input_size + num_non_periodic_input_size,
                categorical_input_size=cat_input_size,
                num_fourier_features=num_fourier_features,
                hidden_size=hidden_size
            ),
            "cnet": ModelChebyshevNet(
                input_size=num_periodic_input_size + num_non_periodic_input_size,
                num_chebyshev_terms=num_chebyshev_terms
            ),
            "tabcnet": ModelTabChebyshevNet(
                continuous_input_size=num_periodic_input_size + num_non_periodic_input_size,
                categorical_input_size=cat_input_size,
                num_chebyshev_terms=num_chebyshev_terms,
                hidden_size=hidden_size
            ),
            "pnpnet": ModelPNPNet(
                periodic_input_size=num_periodic_input_size,
                non_periodic_input_size=num_non_periodic_input_size,
                num_fourier_features=num_fourier_features,
                num_chebyshev_terms=num_chebyshev_terms
            ),
            "tabpnpnet": ModelTabPNPNet(
                periodic_input_size=num_periodic_input_size,
                non_periodic_input_size=num_non_periodic_input_size,
                categorical_input_size=cat_input_size,
                num_fourier_features=num_fourier_features,
                num_chebyshev_terms=num_chebyshev_terms,
                hidden_size=hidden_size
            ),
            "autopnpnet": ModelAutoPNPNet(
                input_size=num_periodic_input_size + num_non_periodic_input_size,
                num_fourier_features=num_fourier_features,
                num_chebyshev_terms=num_chebyshev_terms
            ),
            "tabautopnpnet": ModelTabAutoPNPNet(
                continuous_input_size=num_periodic_input_size + num_non_periodic_input_size,
                categorical_input_size=cat_input_size,
                num_fourier_features=num_fourier_features,
                num_chebyshev_terms=num_chebyshev_terms,
                hidden_size=hidden_size
            )
        }

    def get_model(self, model_name: str) -> nn.Module:
        return self.models[model_name]

from torch import nn

from src.classes.periodicity.models.autopnp.ModelAutoPNPNet import ModelAutoPNPNet
from src.classes.periodicity.models.autopnp.ModelTabAutoPNPNet import ModelTabAutoPNPNet
from src.classes.periodicity.models.chebyshev.ModelChebyshevNet import ModelChebyshevNet
from src.classes.periodicity.models.chebyshev.ModelTabChebyshevNet import ModelTabChebyshevNet
from src.classes.periodicity.models.fourier.ModelFourierNet import ModelFourierNet
from src.classes.periodicity.models.fourier.ModelTabFourierNet import ModelTabFourierNet
from src.classes.periodicity.models.pnp.ModelPNPNet import ModelPNPNet
from src.classes.periodicity.models.pnp.ModelTabPNPNet import ModelTabPNPNet


class ModelFactory:

    def __init__(self, num_periodic_input_size: int, num_non_periodic_input_size: int, cat_input_size: int,
                 num_fourier_features: int = 16, num_chebyshev_terms: int = 5, hidden_size: int = 64,
                 dropout_prob: float = 0.2, batch_norm: bool = True):
        self.num_periodic_input_size = num_periodic_input_size
        self.num_non_periodic_input_size = num_non_periodic_input_size
        self.cat_input_size = cat_input_size
        self.num_fourier_features = num_fourier_features
        self.num_chebyshev_terms = num_chebyshev_terms
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.batch_norm = batch_norm

        self.models = {
            "fnet": ModelFourierNet(
                input_size=num_periodic_input_size + num_non_periodic_input_size,
                num_fourier_features=num_fourier_features,
                hidden_size=hidden_size,
                dropout_prob=dropout_prob,
                batch_norm=batch_norm
            ),
            "tabfnet": ModelTabFourierNet(
                continuous_input_size=num_periodic_input_size + num_non_periodic_input_size,
                categorical_input_size=cat_input_size,
                num_fourier_features=num_fourier_features,
                hidden_size=hidden_size,
                dropout_prob=dropout_prob,
                batch_norm=batch_norm
            ),
            "cnet": ModelChebyshevNet(
                input_size=num_periodic_input_size + num_non_periodic_input_size,
                num_chebyshev_terms=num_chebyshev_terms,
                hidden_size=hidden_size,
                dropout_prob=dropout_prob,
                batch_norm=batch_norm
            ),
            "tabcnet": ModelTabChebyshevNet(
                continuous_input_size=num_periodic_input_size + num_non_periodic_input_size,
                categorical_input_size=cat_input_size,
                num_chebyshev_terms=num_chebyshev_terms,
                hidden_size=hidden_size,
                dropout_prob=dropout_prob,
                batch_norm=batch_norm
            ),
            "pnpnet": ModelPNPNet(
                periodic_input_size=num_periodic_input_size,
                non_periodic_input_size=num_non_periodic_input_size,
                num_fourier_features=num_fourier_features,
                num_chebyshev_terms=num_chebyshev_terms,
                hidden_size=hidden_size,
                dropout_prob=dropout_prob,
                batch_norm=batch_norm
            ),
            "tabpnpnet": ModelTabPNPNet(
                periodic_input_size=num_periodic_input_size,
                non_periodic_input_size=num_non_periodic_input_size,
                categorical_input_size=cat_input_size,
                num_fourier_features=num_fourier_features,
                num_chebyshev_terms=num_chebyshev_terms,
                hidden_size=hidden_size,
                dropout_prob=dropout_prob,
                batch_norm=batch_norm
            ),
            "autopnpnet": ModelAutoPNPNet(
                input_size=num_periodic_input_size + num_non_periodic_input_size,
                num_fourier_features=num_fourier_features,
                num_chebyshev_terms=num_chebyshev_terms,
                hidden_size=hidden_size,
                dropout_prob=dropout_prob,
                batch_norm=batch_norm
            ),
            "tabautopnpnet": ModelTabAutoPNPNet(
                continuous_input_size=num_periodic_input_size + num_non_periodic_input_size,
                categorical_input_size=cat_input_size,
                num_fourier_features=num_fourier_features,
                num_chebyshev_terms=num_chebyshev_terms,
                hidden_size=hidden_size,
                dropout_prob=dropout_prob,
                batch_norm=batch_norm
            )
        }

    def get_model(self, model_name: str) -> nn.Module:
        return self.models[model_name]

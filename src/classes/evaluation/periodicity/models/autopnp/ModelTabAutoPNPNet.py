import torch

from src.classes.evaluation.periodicity.models.autopnp.TabAutoPNPNet import TabAutoPNPNet
from src.config import DEVICE


class ModelTabAutoPNPNet:

    def __init__(
            self,
            continuous_input_size: int,
            categorical_input_size: int,
            num_fourier_features: int = 16,
            num_chebyshev_terms: int = 5,
            hidden_size: int = 64,
            output_size: int = 1
    ):
        self.network = TabAutoPNPNet(
            continuous_input_size=continuous_input_size,
            categorical_input_size=categorical_input_size,
            num_fourier_features=num_fourier_features,
            num_chebyshev_terms=num_chebyshev_terms,
            hidden_size=hidden_size,
            output_size=output_size
        ).to(DEVICE)

    def predict(
            self,
            x_train_num_p_tsr: torch.Tensor,
            x_train_num_np_tsr: torch.Tensor,
            x_train_cat_tsr: torch.Tensor
    ) -> torch.Tensor:
        x_num_tsr = torch.cat([x_train_num_p_tsr, x_train_num_np_tsr], dim=-1)
        return self.network(x_num_tsr, x_train_cat_tsr)

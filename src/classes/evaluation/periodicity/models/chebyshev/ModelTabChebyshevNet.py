import torch

from src.classes.evaluation.periodicity.models.chebyshev.TabChebyshevNet import TabChebyshevNet
from src.config import DEVICE


class ModelTabChebyshevNet:

    def __init__(
            self,
            continuous_input_size: int,
            categorical_input_size: int,
            num_chebyshev_terms: int = 5,
            hidden_size: int = 64
    ):
        self.network = TabChebyshevNet(
            continuous_input_size=continuous_input_size,
            categorical_input_size=categorical_input_size,
            num_chebyshev_terms=num_chebyshev_terms,
            hidden_size=hidden_size
        ).to(DEVICE)

    def predict(
            self,
            x_train_num_p_tsr: torch.Tensor,
            x_train_num_np_tsr: torch.Tensor,
            x_train_cat_tsr: torch.Tensor
    ) -> torch.Tensor:
        x_num_tsr = torch.cat([x_train_num_p_tsr, x_train_num_np_tsr], dim=-1)
        return self.network(x_num_tsr, x_train_cat_tsr)

import torch

from src.classes.evaluation.periodicity.models.fourier.TabFourierNet import TabFourierNet
from src.config import DEVICE


class ModelTabFourierNet:

    def __init__(
            self,
            continuous_input_size: int,
            categorical_input_size: int,
            num_fourier_features: int = 16,
            hidden_size: int = 64
    ):
        self.network = TabFourierNet(
            continuous_input_size=continuous_input_size,
            categorical_input_size=categorical_input_size,
            num_fourier_features=num_fourier_features,
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

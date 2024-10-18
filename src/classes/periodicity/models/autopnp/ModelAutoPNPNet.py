import torch

from src.classes.periodicity.models.autopnp.AutoPNPNet import AutoPNPNet
from src.settings import DEVICE


class ModelAutoPNPNet:

    def __init__(self, input_size, num_fourier_features: int = 16, num_chebyshev_terms: int = 5, hidden_size: int = 64,
                 dropout_prob: float = 0.2, batch_norm: bool = True):
        self.network = AutoPNPNet(
            input_size=input_size,
            num_fourier_features=num_fourier_features,
            num_chebyshev_terms=num_chebyshev_terms,
            hidden_size=hidden_size,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm
        ).to(DEVICE)

    def predict(self, x_train_num_p_tsr: torch.Tensor, x_train_num_np_tsr: torch.Tensor) -> torch.Tensor:
        x_num_tsr = torch.cat([x_train_num_p_tsr, x_train_num_np_tsr], dim=-1)
        return self.network(x_num_tsr)

import torch

from src.classes.periodicity.models.pnp.PNPNet import PNPNet
from src.settings import DEVICE


class ModelPNPNet:

    def __init__(self, periodic_input_size: int, non_periodic_input_size: int, num_fourier_features: int = 16,
                 num_chebyshev_terms: int = 5, hidden_size: int = 64, dropout_prob: float = 0.2,
                 batch_norm: bool = True):
        self.network = PNPNet(
            periodic_input_size=periodic_input_size,
            non_periodic_input_size=non_periodic_input_size,
            num_fourier_features=num_fourier_features,
            num_chebyshev_terms=num_chebyshev_terms,
            hidden_size=hidden_size,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm
        ).to(DEVICE)

    def predict(self, x_train_num_p_tsr: torch.Tensor, x_train_num_np_tsr: torch.Tensor) -> torch.Tensor:
        return self.network(x_train_num_p_tsr, x_train_num_np_tsr)

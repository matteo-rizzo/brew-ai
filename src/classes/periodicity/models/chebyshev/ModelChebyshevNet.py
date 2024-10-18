import torch

from src.classes.periodicity.models.chebyshev.ChebyshevNet import ChebyshevNet
from src.settings import DEVICE


class ModelChebyshevNet:

    def __init__(self, input_size: int, num_chebyshev_terms: int = 5,
                 hidden_size: int = 64, dropout_prob: float = 0.2, batch_norm: bool = True):
        self.network = ChebyshevNet(
            input_size=input_size,
            num_chebyshev_terms=num_chebyshev_terms,
            hidden_size=hidden_size,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm
        ).to(DEVICE)

    def predict(self, x_train_num_p_tsr: torch.Tensor, x_train_num_np_tsr: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_train_num_p_tsr, x_train_num_np_tsr], dim=-1)
        return self.network(x)

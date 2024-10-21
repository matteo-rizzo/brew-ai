import torch

from src.classes.evaluation.periodicity.models.chebyshev.ChebyshevNet import ChebyshevNet
from src.config import DEVICE


class ModelChebyshevNet:

    def __init__(self, input_size: int, num_chebyshev_terms: int = 5):
        self.network = ChebyshevNet(input_size, num_chebyshev_terms).to(DEVICE)

    def predict(self, x_train_num_p_tsr: torch.Tensor, x_train_num_np_tsr: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_train_num_p_tsr, x_train_num_np_tsr], dim=-1)
        return self.network(x)

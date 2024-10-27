import torch

from src.classes.evaluation.periodicity.models.autopnp.AutoPNPNet import AutoPNPNet
from src.config import DEVICE


class ModelAutoPNPNet:

    def __init__(self, input_size, num_fourier_features: int = 16, num_chebyshev_terms: int = 5, output_size: int = 1):
        self.network = AutoPNPNet(input_size, num_fourier_features, num_chebyshev_terms, output_size).to(DEVICE)

    def predict(self, x_train_num_p_tsr: torch.Tensor, x_train_num_np_tsr: torch.Tensor) -> torch.Tensor:
        x_num_tsr = torch.cat([x_train_num_p_tsr, x_train_num_np_tsr], dim=-1)
        return self.network(x_num_tsr)

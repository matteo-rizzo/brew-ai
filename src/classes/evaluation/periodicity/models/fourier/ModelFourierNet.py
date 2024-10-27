import torch

from src.classes.evaluation.periodicity.models.fourier.FourierNet import FourierNet
from src.config import DEVICE


class ModelFourierNet:

    def __init__(self, input_size: int, num_fourier_features: int = 16, output_size: int = 1):
        self.network = FourierNet(input_size, num_fourier_features, output_size).to(DEVICE)

    def predict(self, x_train_num_p_tsr: torch.Tensor, x_train_num_np_tsr: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_train_num_p_tsr, x_train_num_np_tsr], dim=-1)
        return self.network(x)

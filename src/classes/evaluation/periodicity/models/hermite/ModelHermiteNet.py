import torch

from src.classes.evaluation.periodicity.models.hermite.HermiteNet import HermiteNet
from src.config import DEVICE


class ModelHermiteNet:

    def __init__(self, input_size: int, hermite_degree: int = 16):
        self.network = HermiteNet(input_size, hermite_degree).to(DEVICE)

    def predict(self, x_train_num_p_tsr: torch.Tensor, x_train_num_np_tsr: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_train_num_p_tsr, x_train_num_np_tsr], dim=-1)
        return self.network(x)

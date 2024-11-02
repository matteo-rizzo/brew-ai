import torch

from src.classes.evaluation.periodicity.models.pnp.PNPNet import PNPNet
from src.config import DEVICE


class ModelPNPNet:

    def __init__(
            self,
            periodic_input_size: int,
            non_periodic_input_size: int,
            num_fourier_features: int = 16,
            max_poly_terms: int = 5,
            output_size: int = 1
    ):
        self.network = PNPNet(
            periodic_input_size=periodic_input_size,
            non_periodic_input_size=non_periodic_input_size,
            num_fourier_features=num_fourier_features,
            max_poly_terms=max_poly_terms,
            output_size=output_size
        ).to(DEVICE)

    def predict(self, x_train_num_p_tsr: torch.Tensor, x_train_num_np_tsr: torch.Tensor) -> torch.Tensor:
        return self.network(x_train_num_p_tsr, x_train_num_np_tsr)

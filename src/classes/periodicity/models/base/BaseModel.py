from typing import Optional

import torch
from torch import nn

from src.config import DEVICE


class BaseModel:
    def __init__(self, network: Optional[nn.Module] = None):
        """
        BaseModel class with a customizable neural network for predictions.

        :param network: An optional neural network (nn.Module) for performing predictions.
        """
        if network is None:
            raise ValueError("A valid network must be provided.")
        self.network = network.to(DEVICE)

    def predict(self, x_train_num_p_tsr: torch.Tensor, x_train_num_np_tsr: torch.Tensor) -> torch.Tensor:
        """
        Concatenates input tensors and performs prediction using the initialized network.

        :param x_train_num_p_tsr: A tensor of numerical features with processing.
        :param x_train_num_np_tsr: A tensor of numerical features without processing.
        :return: Output tensor after passing through the network.
        :raises ValueError: If network is not initialized before calling predict.
        """
        if self.network is None:
            raise ValueError("Network is not initialized. Please assign a network before calling predict.")

        # Concatenate input tensors along the last dimension
        x = torch.cat([x_train_num_p_tsr, x_train_num_np_tsr], dim=-1)
        return self.network(x)

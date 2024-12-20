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

    def predict(
            self,
            x_num_p: torch.Tensor = None,
            x_num_np: torch.Tensor = None,
            x_num: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Concatenates input tensors and performs prediction using the initialized network.

        :param x_num: A tensor of numerical features. If provided, x_num_p and x_num_np will be ignored.
        :param x_num_p: A tensor of periodic numerical features. Must be provided if x_num is None.
        :param x_num_np: A tensor of non-periodic numerical features. Must be provided if x_num is None.
        :return: Output tensor after passing through the network.
        :raises ValueError: If network is not initialized before calling predict.
        """
        if self.network is None:
            raise ValueError("Network is not initialized. Please assign a network before calling predict.")

        if x_num is not None:
            x = x_num
        elif x_num_p is not None and x_num_np is not None:
            # Concatenate input tensors along the last dimension
            x = torch.cat([x_num_p, x_num_np], dim=-1)
        else:
            raise ValueError("You must either provide x_num_p and x_num_np or x_train_num.")

        return self.network(x)

from typing import Optional

import torch
from torch import nn

from src.config import DEVICE


class BaseTabModel:
    def __init__(self, network: Optional[nn.Module] = None):
        """
        BaseTabModel serves as a base model class for tabular data, supporting both numerical and categorical features.

        :param network: An optional neural network (nn.Module) for processing the data. The network is moved to the specified device.
        """
        if network is None:
            raise ValueError("A valid network must be provided.")
        self.network = network.to(DEVICE)

    def predict(
            self,
            x_num_p: torch.Tensor = None,
            x_num_np: torch.Tensor = None,
            x_cat: torch.Tensor = None,
            x_num: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Concatenates processed and non-processed numerical features, and predicts using the network.

        :param x_num: A tensor of numerical features. If provided, x_num_p and x_num_np will be ignored.
        :param x_num_p: A tensor of periodic numerical features. Must be provided if x_num is None.
        :param x_num_np: A tensor of non-periodic numerical features. Must be provided if x_num is None.
        :param x_cat: Tensor of categorical features.
        :return: Output tensor after passing concatenated features through the network.
        :raises ValueError: If network is not initialized.
        """
        if self.network is None:
            raise ValueError("Network is not initialized. Please assign a network before calling predict.")

        if x_num is not None:
            x_num_tsr = x_num
        elif x_num_p is not None and x_num_np is not None:
            # Concatenate input tensors along the last dimension
            x_num_tsr = torch.cat([x_num_p, x_num_np], dim=-1)
        else:
            raise ValueError("You must either provide x_num_p and x_num_np or x_train_num.")

        # Perform prediction with the network
        return self.network(x_num_tsr, x_cat)

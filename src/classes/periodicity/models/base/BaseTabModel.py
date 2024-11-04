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
            x_train_num_p_tsr: torch.Tensor,
            x_train_num_np_tsr: torch.Tensor,
            x_train_cat_tsr: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenates processed and non-processed numerical features, and predicts using the network.

        :param x_train_num_p_tsr: Tensor of periodical numerical features.
        :param x_train_num_np_tsr: Tensor of non-periodical numerical features.
        :param x_train_cat_tsr: Tensor of categorical features.
        :return: Output tensor after passing concatenated features through the network.
        :raises ValueError: If network is not initialized.
        """
        if self.network is None:
            raise ValueError("Network is not initialized. Please assign a network before calling predict.")

        # Concatenate numerical tensors along the last dimension
        x_num_tsr = torch.cat([x_train_num_p_tsr, x_train_num_np_tsr], dim=-1)

        # Perform prediction with the network
        return self.network(x_num_tsr, x_train_cat_tsr)

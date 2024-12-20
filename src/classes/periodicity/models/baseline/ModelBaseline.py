import torch
from rtdl import FTTransformer

from src.config import DEVICE


class ModelBaseline:

    def __init__(self, dataset_config: dict, output_size: int = 1):
        self.network = FTTransformer.make_baseline(
            n_num_features=dataset_config['n_num_features'],
            cat_cardinalities=dataset_config['cat_cards'],
            d_token=8,
            n_blocks=2,
            attention_dropout=0.2,
            ffn_d_hidden=6,
            ffn_dropout=0.2,
            residual_dropout=0.0,
            d_out=output_size,
        ).to(DEVICE)

    def predict(
            self,
            x_num_p: torch.Tensor,
            x_num_np: torch.Tensor,
            x_cat: torch.Tensor
    ) -> torch.Tensor:
        x_num_tsr = torch.cat([x_num_p, x_num_np], dim=-1)
        if not x_cat.numel():
            x_cat = None
        return self.network(x_num_tsr, x_cat)

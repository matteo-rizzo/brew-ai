import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from rtdl import FTTransformer
from tab_transformer_pytorch import TabTransformer

from src.classes.utils.Logger import Logger
from src.settings import RANDOM_SEED

# Logger setup
logger = Logger()


class ModelFactory:
    """
    Factory class to initialize deep learning models based on the selected strategy.

    Supports the following models:
    - TabNet
    - NODE (Neural Oblivious Decision Ensembles)
    - TabTransformer
    - Wide & Deep (for mixed memorization and generalization)
    """

    def __init__(self, model_name: str):
        """
        Initialize the factory with the model name and optional parameters.

        :param model_name: The name of the model to initialize (e.g., 'tabnet', 'node', 'tabtransformer', 'widedeep')
        """
        self.model_name = model_name.lower()

    def get_model(self):
        """
        Returns an initialized model based on the selected strategy.

        :return: Initialized machine learning model
        :raises ValueError: If the model name is not recognized
        """
        try:
            logger.info(f"Initializing model: {self.model_name}")

            if self.model_name == "tabnet":
                # TabNet Regressor
                return TabNetRegressor(
                    cat_idxs=list(range(24, 28)),
                    cat_dims=[2, 2, 2, 2],
                    scheduler_params={"step_size": 10, "gamma": 0.9},
                    scheduler_fn=torch.optim.lr_scheduler.StepLR,
                    verbose=1,
                    seed=RANDOM_SEED
                )

            elif self.model_name == "fttransformer":
                # Neural Oblivious Decision Ensembles (NODE)
                return FTTransformer.make_baseline(
                    n_num_features=23,
                    cat_cardinalities=[2, 2, 2, 2],
                    d_token=8,
                    n_blocks=2,
                    attention_dropout=0.2,
                    ffn_d_hidden=6,
                    ffn_dropout=0.2,
                    residual_dropout=0.0,
                    d_out=1,
                )

            elif self.model_name == "tabtransformer":
                # TabTransformer
                return TabTransformer(
                    categories=[2, 2, 2, 2],
                    num_continuous=23,
                    dim=32,
                    depth=6,
                    heads=8,
                    attn_dropout=0.1,
                    ff_dropout=0.1,
                )
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

        except Exception as e:
            logger.error(f"Error initializing model {self.model_name}: {e}")
            raise

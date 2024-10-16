from typing import Tuple, Union

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from rtdl import FTTransformer
from sklearn.base import BaseEstimator
from tab_transformer_pytorch import TabTransformer

from src.classes.utils.Logger import Logger

logger = Logger()


class Evaluator:
    """
    Evaluates trained models on a test set using common regression metrics.
    Supports TabNet, FTTransformer, TabTransformer, and other models.
    """

    def __init__(self, model: Union[torch.nn.Module, BaseEstimator], idx_num: list, idx_cat: list):
        self.model = model
        self.idx_num = idx_num
        self.idx_cat = idx_cat

    def evaluate(self, x_test: np.ndarray) -> np.ndarray:
        """
        Evaluate the model on the test set.
        """
        if isinstance(self.model, TabNetRegressor):
            return self._evaluate_tabnet(x_test)
        elif isinstance(self.model, FTTransformer):
            return self._evaluate_pytorch_model(x_test, model_type="FTTransformer")
        elif isinstance(self.model, TabTransformer):
            return self._evaluate_pytorch_model(x_test, model_type="TabTransformer")
        else:
            raise ValueError(f"Unsupported model type: {self.model.__class__.__name__}")

    def _split_vars(self, x: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split the input into numerical and categorical feature tensors.
        """
        x_num_tensor = torch.tensor(x[:, self.idx_num], dtype=torch.float32)
        x_cat_tensor = torch.tensor(x[:, self.idx_cat], dtype=torch.long)
        return x_num_tensor, x_cat_tensor

    def _evaluate_tabnet(self, x_test: np.ndarray) -> np.ndarray:
        """Evaluate TabNet model."""
        logger.info("Evaluating TabNet model...")
        return self.model.predict(x_test)

    def _evaluate_pytorch_model(self, x_test: np.ndarray, model_type: str) -> np.ndarray:
        """
        Evaluate a PyTorch-based model (FTTransformer or TabTransformer).
        """
        logger.info(f"Evaluating {model_type} model...")

        # Convert input arrays to PyTorch tensors
        x_test_num_tensor, x_test_cat_tensor = self._split_vars(x_test)

        # Set model to evaluation mode and disable gradient tracking
        self.model.eval()
        with torch.no_grad():
            if model_type == "FTTransformer":
                predictions = self.model(x_test_num_tensor, x_test_cat_tensor)
            else:
                # TabTransformer takes categorical inputs first
                predictions = self.model(x_test_cat_tensor, x_test_num_tensor)

        return predictions.numpy()

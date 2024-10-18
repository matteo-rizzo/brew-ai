from typing import Tuple, List, Union

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from rtdl import FTTransformer
from sklearn.base import BaseEstimator
from tab_transformer_pytorch import TabTransformer
from torch import nn

from src.classes.utils.Logger import Logger
from src.settings import EPOCHS, LR

logger = Logger()


class Trainer:
    """
    Handles training for various types of deep_learning learning and machine learning models.
    Supports TabNet, FTTransformer, TabTransformer, and other models.
    """

    def __init__(self, model: Union[torch.nn.Module, BaseEstimator], idx_num: List[int], idx_cat: List[int],
                 epochs: int = EPOCHS, learning_rate: float = LR):
        self.model = model
        self.idx_num = idx_num
        self.idx_cat = idx_cat
        self.epochs = epochs
        self.learning_rate = learning_rate

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model on a single fold.
        """
        y_train = y_train.reshape(-1, 1)

        if isinstance(self.model, TabNetRegressor):
            self._train_tabnet(x_train, y_train)
        elif isinstance(self.model, FTTransformer):
            self._train_pytorch_model(x_train, y_train, model_type="FTTransformer")
        elif isinstance(self.model, TabTransformer):
            self._train_pytorch_model(x_train, y_train, model_type="TabTransformer")
        else:
            raise ValueError(f"Unsupported model type: {self.model.__class__.__name__}")

    def _convert_to_tensor(self, x: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert numpy arrays to PyTorch tensors for numerical and categorical inputs.
        """
        x_num_tensor = torch.tensor(x[:, self.idx_num], dtype=torch.float32)
        x_cat_tensor = torch.tensor(x[:, self.idx_cat], dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_num_tensor, x_cat_tensor, y_tensor

    def _train_tabnet(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train TabNet model."""
        logger.info("Training TabNet model...")
        self.model.fit(
            X_train=x_train,
            y_train=y_train,
            max_epochs=self.epochs
        )
        logger.info("TabNet training completed.")

    def _train_pytorch_model(self, x_train: np.ndarray, y_train: np.ndarray, model_type: str) -> None:
        """Train PyTorch-based models like FTTransformer or TabTransformer."""
        logger.info(f"Training {model_type} model...")

        # Convert numpy arrays to PyTorch tensors
        x_train_num_tensor, x_train_cat_tensor, y_train_tensor = self._convert_to_tensor(x_train, y_train)

        if model_type == "FTTransformer":
            self._pytorch_train_loop(x_train_num_tensor, x_train_cat_tensor, y_train_tensor)
        else:
            self._pytorch_train_loop(x_train_cat_tensor, x_train_num_tensor, y_train_tensor)

        logger.info(f"{model_type} training completed.")

    def _pytorch_train_loop(self, x1_train: torch.Tensor, x2_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """
        General PyTorch training loop for models like FTTransformer and TabTransformer.
        x1_train and x2_train represent either numerical or categorical inputs depending on the model.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()

            # Model expects inputs in specific order
            predictions = self.model(x1_train, x2_train)

            # Compute the loss
            loss = loss_fn(predictions, y_train)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:  # Log progress every 10 epochs
                logger.info(f"Epoch {epoch}/{self.epochs} - Loss: {loss.item()}")

    def get_model(self):
        return self.model

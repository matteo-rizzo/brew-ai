from typing import Dict, Tuple, List, Union

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.classes.periodicity.models.base.BaseModel import BaseModel
from src.classes.periodicity.models.base.BaseTabModel import BaseTabModel
from src.classes.utils.Logger import Logger
from src.config import DEVICE, CLASSIFICATION

logger = Logger()


class Trainer:
    def __init__(
            self,
            model: Union[BaseModel, BaseTabModel],
            model_name: str,
            criterion,
            optimizer,
            batch_size: int,
            num_epochs: int,
            patience: int,
            scheduler_patience: int = 10,
            early_stopping: bool = True,
    ):
        """
        Trainer class to handle model training and validation.

        :param model: The model to be trained.
        :param model_name: The name of the model to be trained.
        :param criterion: Loss function.
        :param optimizer: Optimizer for model parameters.
        :param batch_size: Batch size for training.
        :param num_epochs: Number of epochs for training.
        :param early_stopping: Whether to apply early stopping based on validation loss.
        :param patience: Number of epochs to wait for improvement in validation loss before stopping early.
        :param scheduler_patience: Patience for learning rate scheduler before reducing learning rate.
        """
        self.model = model
        self.model_name = model_name
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.device = DEVICE

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=scheduler_patience, factor=0.1
        )

    def train(self, train_data: Dict[str, torch.Tensor], val_data: Dict[str, torch.Tensor]) -> Tuple[
        List[float], List[float]]:
        """
        Train the model and validate after each epoch.

        :param train_data: A dictionary containing the training data tensors.
        :param val_data: A dictionary containing the validation data tensors.
        :return: Lists of training and validation losses per epoch.
        """
        x_train_num_p, x_train_num_np, x_train_cat, y_train = self._unpack_data(train_data)
        x_val_num_p, x_val_num_np, x_val_cat, y_val = self._unpack_data(val_data)

        # Create DataLoader for training data
        train_loader = self._create_dataloader(x_train_num_p, x_train_num_np, x_train_cat, y_train)

        # Lists to store losses
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(self.num_epochs):
            train_loss = self._train_one_epoch(train_loader)
            train_losses.append(train_loss)

            # Validation
            val_loss = self._validate(x_val_num_p, x_val_num_np, x_val_cat, y_val)
            val_losses.append(val_loss)

            # Step the scheduler based on the validation loss
            self.scheduler.step(val_loss)

            # Logging progress
            logger.info(
                f"Model: {self.model_name} - Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

            # Check for early stopping
            if self.early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0  # Reset patience counter if improvement
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
                        break

        return train_losses, val_losses

    def _train_one_epoch(self, train_loader: DataLoader) -> float:
        """
        Train the model for one epoch.

        :param train_loader: DataLoader for training data.
        :return: The average training loss for the epoch.
        """
        self.model.network.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch_x_p, batch_x_np, batch_x_cat, batch_y = batch
            batch_x_p, batch_x_np, batch_x_cat, batch_y = (
                batch_x_p.to(self.device), batch_x_np.to(self.device), batch_x_cat.to(self.device),
                batch_y.to(self.device)
            )

            self.optimizer.zero_grad()
            outputs = self._predict(batch_x_p, batch_x_np, batch_x_cat)
            if CLASSIFICATION:
                batch_y = batch_y.squeeze().long()
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * batch_x_p.size(0)

        return epoch_loss / len(train_loader.dataset)

    def _validate(self, x_val_num_p: torch.Tensor, x_val_num_np: torch.Tensor, x_val_cat: torch.Tensor,
                  y_val: torch.Tensor) -> float:
        """
        Validate the model using the validation dataset.

        :param x_val_num_p: Periodic numerical validation data.
        :param x_val_num_np: Non-periodic numerical validation data.
        :param x_val_cat: Categorical validation data.
        :param y_val: Target validation data.
        :return: Validation loss.
        """
        self.model.network.eval()
        with torch.no_grad():
            x_val_num_p, x_val_num_np, x_val_cat, y_val = (
                x_val_num_p.to(self.device), x_val_num_np.to(self.device), x_val_cat.to(self.device),
                y_val.to(self.device)
            )
            outputs = self._predict(x_val_num_p, x_val_num_np, x_val_cat)
            if CLASSIFICATION:
                y_val = y_val.squeeze().long()
            val_loss = self.criterion(outputs, y_val)
        return val_loss.item()

    def _predict(self, x_num_p: torch.Tensor, x_num_np: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Predict using the model based on the input data.

        :param x_num_p: Periodic numerical data.
        :param x_num_np: Non-periodic numerical data.
        :param x_cat: Categorical data.
        :return: Model predictions.
        """
        if self.model_name.startswith("tab"):
            return self.model.predict(x_num_p, x_num_np, x_cat)
        else:
            return self.model.predict(x_num_p, x_num_np)

    @staticmethod
    def _unpack_data(data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unpack the training or validation data.

        :param data: Dictionary containing input tensors.
        :return: Unpacked tensors for periodic numerical, non-periodic numerical, categorical, and target data.
        """
        return data['x_num_p'], data['x_num_np'], data['x_cat'], data['y']

    def _create_dataloader(self, x_num_p: torch.Tensor, x_num_np: torch.Tensor, x_cat: torch.Tensor,
                           y: torch.Tensor) -> DataLoader:
        """
        Create DataLoader from the input data.

        :param x_num_p: Periodic numerical input data.
        :param x_num_np: Non-periodic numerical input data.
        :param x_cat: Categorical input data.
        :param y: Target data.
        :return: DataLoader for training.
        """
        dataset = TensorDataset(x_num_p, x_num_np, x_cat, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def get_model(self) -> Union[BaseModel, BaseTabModel]:
        """
        Get the trained model.

        :return: The trained model.
        """
        return self.model

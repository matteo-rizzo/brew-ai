import os
import time
from typing import Tuple, Dict, List, Union

import pandas as pd
import torch
from torch import nn, optim

from src.classes.data.DataSplitter import DataSplitter
from src.classes.periodicity.ModelFactory import ModelFactory
from src.classes.periodicity.Trainer import Trainer
from src.classes.periodicity.loss.PNPLoss import PNPMSELoss
from src.classes.periodicity.models.base.BaseModel import BaseModel
from src.classes.periodicity.models.base.BaseTabModel import BaseTabModel
from src.classes.utils.Logger import Logger
from src.classes.utils.Plotter import Plotter
from src.config import CLASSIFICATION, PATIENCE

logger = Logger()


class ProdHandler:
    def __init__(
            self,
            model_name: str,
            dataset_config: dict,
            x: pd.DataFrame,
            y: pd.Series,
            idx_num: List[int],
            idx_cat: List[int],
            idx_periodic: List[int],
            idx_non_periodic: List[int],
            batch_size: int,
            num_epochs: int,
            learning_rate: float,
            log_dir: str
    ):
        """
        ProdHandler class to handle model training and validation.

        :param model_name: Name of the model being trained.
        :param dataset_config: Configuration dictionary for dataset parameters.
        :param x: Feature matrix as a Pandas DataFrame.
        :param y: Target labels or values as a Pandas Series.
        :param idx_num: Indices for numerical features.
        :param idx_cat: Indices for categorical features.
        :param idx_periodic: Indices for periodic features.
        :param idx_non_periodic: Indices for non-periodic features.
        :param batch_size: Batch size for training.
        :param num_epochs: Number of training epochs.
        :param learning_rate: Learning rate for the optimizer.
        :param log_dir: Directory for saving logs and trained models.
        """
        self.model_name = model_name
        self.dataset_config = dataset_config
        self.x = x
        self.y = y
        self.idx_num = idx_num
        self.idx_cat = idx_cat
        self.idx_periodic = idx_periodic
        self.idx_non_periodic = idx_non_periodic
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.log_dir = log_dir

    def run(self) -> None:
        """
        Run the model training pipeline and save the final model.
        """
        logger.info("Starting model training pipeline.")

        # Step 1: Split the data
        split_data, input_sizes, output_size = self._split_data()

        # Step 2: Initialize the model
        model = self._initialize_model(input_sizes, output_size)

        # Step 3: Train the model
        trained_model, train_losses, val_losses = self._train_model(model, split_data)

        # Step 4: Plot and save losses
        logger.info("Plotting and saving loss curves.")
        Plotter.plot_and_save_losses(train_losses, val_losses, -1, self.log_dir)

        logger.info("Model training pipeline completed successfully.")

    def _split_data(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], int]:
        """
        Split the dataset into training and validation sets.

        :return: A tuple containing the split data, input sizes, and output size.
        """
        logger.info("Splitting data into training and validation sets.")

        splitter = DataSplitter(
            self.x,
            self.y,
            self.idx_num,
            self.idx_cat,
            self.idx_periodic,
            self.idx_non_periodic
        )

        split_data = splitter.split()
        input_sizes = split_data['input_sizes']
        output_size = self.y.nunique() if CLASSIFICATION else 1

        logger.info("Data splitting completed.")
        return split_data, input_sizes, output_size

    def _initialize_model(self, input_sizes: Dict[str, int], output_size: int) -> Union[BaseModel, BaseTabModel]:
        """
        Initialize the model using the ModelFactory.

        :param input_sizes: Input sizes for model components.
        :param output_size: Output size for classification or regression.
        :return: Initialized PyTorch model.
        """
        logger.info("Initializing the model using ModelFactory.")

        model_factory = ModelFactory(
            num_periodic_input_size=input_sizes['num_periodic_input_size'],
            num_non_periodic_input_size=input_sizes['num_non_periodic_input_size'],
            cat_input_size=input_sizes['cat_input_size'],
            output_size=output_size,
            dataset_config=self.dataset_config
        )

        model = model_factory.get_model(self.model_name)
        logger.info(f"Model '{self.model_name}' initialized successfully.")
        return model

    def _train_model(
            self,
            model: Union[BaseModel, BaseTabModel],
            split_data: Dict[str, torch.Tensor]
    ) -> tuple[BaseModel | BaseTabModel, list[float], list[float]]:
        """
        Train the model using the Trainer class.

        :param model: The initialized PyTorch model.
        :param split_data: Dictionary with training and validation data.
        :return: Trained model, training losses, and validation losses.
        """
        logger.info("Starting model training.")
        start_time = time.time()

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss() if CLASSIFICATION else PNPMSELoss()
        optimizer = optim.AdamW(model.network.parameters(), lr=self.learning_rate)

        # Trainer setup
        trainer = Trainer(
            model=model,
            model_name=self.model_name,
            criterion=criterion,
            optimizer=optimizer,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            patience=PATIENCE
        )

        # Training process
        train_losses, val_losses = trainer.train(split_data['train'], split_data['val'])
        trained_model = trainer.get_model()

        # Save the trained model
        model_path = os.path.join(self.log_dir, f'{self.model_name}.pth')
        torch.save(trained_model.network.state_dict(), model_path)  # Save the entire model for production
        logger.info(f"Model saved to {model_path}")

        train_duration = time.time() - start_time
        logger.info(f"Training completed in {train_duration:.2f} seconds.")
        return trained_model, train_losses, val_losses

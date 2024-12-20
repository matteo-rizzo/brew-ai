import os.path
import time
from typing import Tuple, Dict, List, Union

import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn, optim

from src.classes.data.DataSplitter import DataSplitter
from src.classes.periodicity.Evaluator import Evaluator
from src.classes.periodicity.ModelFactory import ModelFactory
from src.classes.periodicity.Trainer import Trainer
from src.classes.periodicity.loss.PNPLoss import PNPMSELoss
from src.classes.periodicity.models.base.BaseModel import BaseModel
from src.classes.periodicity.models.base.BaseTabModel import BaseTabModel
from src.classes.utils.Logger import Logger
from src.classes.utils.MetricsCalculator import MetricsCalculator
from src.classes.utils.Plotter import Plotter
from src.config import DEVICE, RANDOM_SEED, CLASSIFICATION, PATIENCE

logger = Logger()


class CrossValidator:
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
            num_folds: int,
            batch_size: int,
            num_epochs: int,
            learning_rate: float,
            log_dir: str
    ):
        """
        CrossValidator class to handle k-fold cross-validation.

        :param model_name: Name of the model being validated.
        :param x: Feature matrix as a Pandas DataFrame.
        :param y: Labels as a Pandas Series.
        :param idx_num: List of indices for numerical features.
        :param idx_cat: List of indices for categorical features.
        :param idx_periodic: List of indices for periodic numerical features.
        :param idx_non_periodic: List of indices for non-periodic numerical features.
        :param num_folds: Number of folds for cross-validation.
        :param batch_size: Batch size for training.
        :param num_epochs: Number of epochs for training.
        :param learning_rate: Learning rate for the optimizer.
        :param log_dir: Directory for saving logs and metrics.
        """
        self.model_name = model_name
        self.dataset_config = dataset_config
        self.x = x
        self.y = y
        self.idx_num = idx_num
        self.idx_cat = idx_cat
        self.idx_periodic = idx_periodic
        self.idx_non_periodic = idx_non_periodic
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.log_dir = log_dir

    def run(self) -> None:
        """
        Run k-fold cross-validation and log metrics for each fold.
        """
        logger.info(f"Initializing {self.num_folds}-fold cross-validation for model: {self.model_name}")

        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=RANDOM_SEED)
        cv_metrics = []

        for fold, (train_val_index, test_index) in enumerate(kf.split(self.x)):
            logger.info(f"\nStarting Fold {fold + 1}/{self.num_folds}")

            fold_start_time = time.time()

            # Split data
            split_data, test_data, input_sizes, output_size = self._split_data(train_val_index, test_index)

            # Initialize model
            model = self._initialize_model(input_sizes, output_size)

            # Train model
            trained_model, train_losses, val_losses = self._train_model(model, split_data, fold)

            # Evaluate model
            evaluator = Evaluator(trained_model, self.model_name, self.log_dir)
            metrics = evaluator.evaluate(test_data, fold)

            # Log fold completion
            fold_duration = time.time() - fold_start_time
            logger.info(f"Fold {fold + 1} completed in {fold_duration:.2f} seconds.")
            logger.info(f"Metrics for Fold {fold + 1}: {metrics}")

            cv_metrics.append(metrics)

            # Plot and save losses
            Plotter.plot_and_save_losses(train_losses, val_losses, fold, self.log_dir)

        # Compute and log average metrics
        self._compute_and_log_avg_metrics(cv_metrics)

    def _split_data(
            self,
            train_val_index: List[int],
            test_index: List[int]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, int], int]:
        """
        Split the data into training, validation, and test sets.

        :param train_val_index: Indices for training/validation split.
        :param test_index: Indices for the test split.
        :return: Tuple containing split data (train/val), test data, and input sizes.
        """
        logger.info("Splitting data into training/validation and test sets.")

        x_train_val, x_test = self.x[train_val_index], self.x[test_index]
        y_train_val, y_test = self.y[train_val_index], self.y[test_index]

        splitter = DataSplitter(
            x_train_val,
            y_train_val,
            self.idx_num,
            self.idx_cat,
            self.idx_periodic,
            self.idx_non_periodic
        )

        split_data = splitter.split()
        test_data = {
            'x_num_p': torch.tensor(x_test[:, self.idx_num][:, self.idx_periodic], dtype=torch.float32).to(DEVICE),
            'x_num_np': torch.tensor(x_test[:, self.idx_num][:, self.idx_non_periodic], dtype=torch.float32).to(DEVICE),
            'x_cat': torch.tensor(x_test[:, self.idx_cat], dtype=torch.int8).to(DEVICE),
            'y': torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        }
        input_sizes = split_data['input_sizes']
        output_size = self.y.nunique() if CLASSIFICATION else 1

        return split_data, test_data, input_sizes, output_size

    def _initialize_model(self, input_sizes: Dict[str, int], output_size: int) -> BaseModel | BaseTabModel:
        """
        Initialize the model using ModelFactory.

        :param input_sizes: Dictionary containing the input sizes for periodic, non-periodic, and categorical data.
        :param input_sizes: Output size, 1 for regression, > 1 for classification.
        :return: Initialized model as a PyTorch Module.
        """
        logger.info("Initializing model with ModelFactory.")

        model_factory = ModelFactory(
            num_periodic_input_size=input_sizes['num_periodic_input_size'],
            num_non_periodic_input_size=input_sizes['num_non_periodic_input_size'],
            cat_input_size=input_sizes['cat_input_size'],
            output_size=output_size,
            dataset_config=self.dataset_config
        )
        model = model_factory.get_model(self.model_name)

        logger.info(f"Model initialization completed.")
        return model

    def _train_model(
            self,
            model: Union[BaseModel, BaseTabModel],
            split_data: Dict[str, torch.Tensor],
            fold: int
    ) -> tuple[BaseModel | BaseTabModel, list[float], list[float]]:
        """
        Train the model using the Trainer class.

        :param model: Initialized model to be trained.
        :param split_data: Dictionary containing the training and validation data.
        :param fold: Index of the current fold.
        :return: A tuple containing the trained model, training losses, and validation losses.
        """
        logger.info("Starting model training.")
        train_start_time = time.time()

        criterion = nn.CrossEntropyLoss() if CLASSIFICATION else PNPMSELoss()
        optimizer = optim.AdamW(model.network.parameters(), lr=self.learning_rate)

        trainer = Trainer(
            model=model,
            model_name=self.model_name,
            criterion=criterion,
            optimizer=optimizer,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            patience=PATIENCE
        )

        train_losses, val_losses = trainer.train(split_data['train'], split_data['val'])
        trained_model = trainer.get_model()
        torch.save(trained_model.network.state_dict(), os.path.join(self.log_dir, f'{self.model_name}_fold_{fold}.pth'))

        train_duration = time.time() - train_start_time
        logger.info(f"Training completed in {train_duration:.2f} seconds.")
        return trained_model, train_losses, val_losses

    def _compute_and_log_avg_metrics(self, cv_metrics: List[Dict[str, float]]) -> None:
        """
        Compute and log the average metrics across all folds.

        :param cv_metrics: List of metrics for each fold.
        """
        logger.info("Computing and logging average metrics across all folds.")

        avg_metrics = pd.DataFrame(cv_metrics).mean().to_dict()
        MetricsCalculator.save_metrics_to_json(self.model_name, avg_metrics, self.log_dir)

        logger.info(f"Average metric calculation completed.")
        logger.info(f"{self.num_folds}-fold cross-validation completed successfully.")
        logger.info(f"Average Metrics: {avg_metrics}")

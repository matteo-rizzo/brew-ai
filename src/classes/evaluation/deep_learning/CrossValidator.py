from typing import Dict, List

import numpy as np
import pandas as pd
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from src.classes.evaluation.deep_learning.Evaluator import Evaluator
from src.classes.utils.MetricsCalculator import MetricsCalculator
from src.classes.evaluation.deep_learning.Trainer import Trainer
from src.classes.utils.Logger import Logger
from src.config import RANDOM_SEED

logger = Logger()


class CrossValidator:
    """
    Handles training for any model, including cross-validation and test set evaluation.
    Supports both TabNet and traditional models.
    """

    def __init__(
            self,
            model: BaseEstimator,
            x: np.ndarray,
            idx_num: List[int],
            idx_cat: List[int],
            y: np.ndarray,
            log_dir: str,
            n_splits: int = 5,
    ):
        """
        Initialize the CrossValidator with model and data.

        :param model: The machine learning model to cross-validate.
        :param x: Feature matrix.
        :param idx_num: Indices of numerical features.
        :param idx_cat: Indices of categorical features.
        :param y: Target vector.
        :param log_dir: Directory to save logs and metrics.
        :param n_splits: Number of cross-validation folds.
        """
        self.model = model
        self.model_name = self.model.__class__.__name__
        self.idx_num = idx_num
        self.idx_cat = idx_cat
        self.x = x
        self.y = y
        self.log_dir = log_dir
        self.n_splits = n_splits

    def cross_validate(self) -> None:
        """
        Perform cross-validation on the model.
        """
        logger.info(f"Starting {self.n_splits}-fold cross-validation for {self.model_name}...")

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_SEED)
        cv_metrics = []

        progress_columns = [
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
        ]

        with Progress(*progress_columns) as progress:
            task = progress.add_task(f"Cross-validating {self.model_name}", total=self.n_splits)

            for fold, (train_index, val_index) in enumerate(kf.split(self.x), 1):
                logger.info(f"Starting fold {fold}/{self.n_splits}...")

                # Split data into training and validation sets
                x_train_fold, x_val_fold = self.x[train_index], self.x[val_index]
                y_train_fold, y_val_fold = self.y[train_index], self.y[val_index]

                # Train the model on the current fold
                trained_model = self._train_on_fold(fold, x_train_fold, y_train_fold)

                # Evaluate the model after training the fold
                metrics = self._evaluate_on_val_set(fold, trained_model, x_val_fold, y_val_fold)

                logger.info(f"Fold {fold}/{self.n_splits} completed. Metrics: {metrics}")

                cv_metrics.append(metrics)
                progress.update(task, advance=1)

        avg_metrics = pd.DataFrame(cv_metrics).mean().to_dict()
        MetricsCalculator.save_metrics_to_json(self.model_name, avg_metrics, self.log_dir)

        logger.info(f"{self.n_splits}-fold cross-validation completed successfully.")
        logger.info(f"Average Metrics: {avg_metrics}")

    def _train_on_fold(self, fold: int, x_train_fold: np.ndarray, y_train_fold: np.ndarray) -> BaseEstimator:
        """
        Train the model for a single fold.

        :param fold: Current fold number.
        :param x_train_fold: Training features for the current fold.
        :param y_train_fold: Training targets for the current fold.
        :return: Trained model.
        """
        logger.info(f"Training fold {fold} with {self.model_name}...")
        trainer = Trainer(self.model, self.idx_num, self.idx_cat)
        trainer.train(x_train_fold, y_train_fold)
        return trainer.get_model()

    def _evaluate_on_val_set(self, fold: int, model: BaseEstimator, x_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Evaluate the trained model on the validation set after each fold.

        :param fold: Current fold number.
        :param model: Trained model to evaluate.
        :param x_val: Validation features for the current fold.
        :param y_val: Validation targets for the current fold.
        :return: Dictionary of evaluation metrics.
        """
        logger.info(f"Evaluating model on validation set after fold {fold}...")
        evaluator = Evaluator(model, self.idx_num, self.idx_cat)
        y_pred = evaluator.evaluate(x_val)
        metrics = MetricsCalculator.calculate_metrics(self.model_name, y_val, y_pred, self.log_dir, fold)
        return metrics

import numpy as np
from rich.progress import Progress
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, train_test_split

from src.classes.evaluation.MetricsCalculator import MetricsCalculator
from src.classes.evaluation.deep.Evaluator import Evaluator
from src.classes.evaluation.deep.Trainer import Trainer
from src.classes.utils.Logger import Logger
from src.settings import TEST_SIZE, RANDOM_SEED, MODEL_DEEP

logger = Logger()


class CrossValidator:
    """
    Handles training for any model, including cross-validation and test set evaluation.
    Supports both TabNet and traditional models.
    """

    def __init__(self, model: BaseEstimator, x: np.ndarray, idx_num: list, idx_cat: list, y: np.ndarray, log_dir: str,
                 n_splits: int = 5):
        self.model = model
        self.idx_num = idx_num
        self.idx_cat = idx_cat
        self.x = x
        self.y = y
        self.log_dir = log_dir
        self.n_splits = n_splits
        self.test_size = TEST_SIZE

        # Split the data into training and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=self.test_size, random_state=RANDOM_SEED
        )

    def cross_validate(self) -> None:
        """
        Perform cross-validation on the model.
        """
        logger.info(f"Starting {self.n_splits}-fold cross-validation for {self.model.__class__.__name__}...")

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_SEED)

        with Progress() as progress:
            task = progress.add_task(f"Cross-validating {self.model.__class__.__name__}...", total=self.n_splits)

            for fold, (train_index, val_index) in enumerate(kf.split(self.x_train), 1):
                logger.info(f"Starting fold {fold}/{self.n_splits}...")

                # Split data into training and validation sets
                x_train_fold, x_val_fold = self.x_train[train_index], self.x_train[val_index]
                y_train_fold, y_val_fold = self.y_train[train_index], self.y_train[val_index]

                try:
                    # Train the model on the current fold
                    trained_model = self._train_on_fold(fold, x_train_fold, y_train_fold, x_val_fold, y_val_fold)

                    # Evaluate the model after training the fold
                    self._evaluate_on_test_set(fold, trained_model)
                    logger.info(f"Fold {fold}/{self.n_splits} completed successfully.")

                except Exception as e:
                    logger.error(f"Error during fold {fold}: {str(e)}")

                progress.update(task, advance=1)

        logger.info(f"{self.n_splits}-fold cross-validation completed successfully.")

    def _train_on_fold(self, fold: int, x_train_fold: np.ndarray, y_train_fold: np.ndarray,
                       x_val_fold: np.ndarray, y_val_fold: np.ndarray) -> BaseEstimator:
        """
        Train the model for a single fold.
        """
        logger.info(f"Training fold {fold} with {self.model.__class__.__name__}...")
        trainer = Trainer(self.model, self.idx_num, self.idx_cat)
        trainer.train(x_train_fold, y_train_fold, x_val_fold, y_val_fold)
        return trainer.get_model()

    def _evaluate_on_test_set(self, fold: int, model: BaseEstimator) -> None:
        """
        Evaluate the trained model on the test set after each fold.
        """
        logger.info(f"Evaluating model on the test set after fold {fold}...")
        y_pred = Evaluator(model, self.idx_num, self.idx_cat).evaluate(self.x_test)
        MetricsCalculator.evaluate_model(MODEL_DEEP, self.y_test, y_pred, self.log_dir)

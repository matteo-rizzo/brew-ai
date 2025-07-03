import time
from typing import List, Dict

import numpy as np

from src.classes.periodicity.CrossValidator import CrossValidator
from src.classes.periodicity.prod.ProdHandler import ProdHandler
from src.classes.utils.Logger import Logger
from src.config import NUM_FOLDS, EPOCHS, LR, BATCH_SIZE

logger = Logger()


class ExperimentHandler:
    """
    ExperimentHandler class, refactored to be a focused training runner.
    This class is responsible for executing the training pipeline (either Cross-Validation or Production)
    on data that has already been preprocessed.
    The orchestration logic (loading, splitting, preprocessing, saving the preprocessor)
    should now reside in the main training script.
    """

    def __init__(self, model_name: str, dataset_config: Dict, log_dir: str, mode: str = "cv"):
        """
        :param model_name: Name of the model being used in the experiment.
        :param dataset_config: Configuration dictionary for the dataset.
        :param log_dir: Directory for saving logs and results.
        :param mode: Type of experiment, "cv" or "prod".
        """
        self.model_name = model_name
        self.dataset_config = dataset_config
        self.log_dir = log_dir
        self.mode = mode
        logger.info(f"Experiment Handler initialized for model: {model_name} in '{mode}' mode.")

    def run_experiment(
            self,
            x: np.ndarray,
            y: np.ndarray,
            idx_num: List[int],
            idx_cat: List[int],
            idx_periodic: List[int],
            idx_non_periodic: List[int]
    ):
        """
        Run the full experimental pipeline on pre-split and preprocessed data.

        :param x: Preprocessed feature matrix (NumPy array).
        :param y: Labels (NumPy array).
        :param idx_num: Indices of numerical features in the preprocessed array.
        :param idx_cat: Indices of categorical features in the preprocessed array.
        :param idx_periodic: Indices of periodic numerical features in the preprocessed array.
        :param idx_non_periodic: Indices of non-periodic numerical features in the preprocessed array.
        """
        try:
            logger.info("Starting the training process.")

            if self.mode == "cv":
                self.run_cross_validation(x, y, idx_periodic, idx_non_periodic, idx_num, idx_cat)
            elif self.mode == "prod":
                self.run_production(x, y, idx_periodic, idx_non_periodic, idx_num, idx_cat)

            logger.info("Experiment completed successfully.")
        except Exception as e:
            logger.error(f"An error occurred during the experiment: {str(e)}")

    def run_cross_validation(
            self,
            x: np.ndarray,
            y: np.ndarray,
            idx_periodic: List[int],
            idx_non_periodic: List[int],
            idx_num: List[int],
            idx_cat: List[int]
    ):
        """
        Set up and run cross-validation on the model.
        """
        logger.info(f"Starting cross-validation for model: {self.model_name} with configuration: "
                    f"Folds={NUM_FOLDS}, Batch Size={BATCH_SIZE}, Epochs={EPOCHS}, Learning Rate={LR}")

        cross_validator = CrossValidator(
            model_name=self.model_name,
            dataset_config=self.dataset_config,
            x=x,
            y=y,
            idx_num=idx_num,
            idx_cat=idx_cat,
            idx_periodic=idx_periodic,
            idx_non_periodic=idx_non_periodic,
            num_folds=NUM_FOLDS,
            batch_size=BATCH_SIZE,
            num_epochs=EPOCHS,
            learning_rate=LR,
            log_dir=self.log_dir
        )

        start_time = time.time()
        cross_validator.run()
        elapsed_time = time.time() - start_time
        logger.info(f"Cross-validation completed in {elapsed_time / 60:.2f} minutes.")

    def run_production(
            self,
            x: np.ndarray,
            y: np.ndarray,
            idx_periodic: List[int],
            idx_non_periodic: List[int],
            idx_num: List[int],
            idx_cat: List[int]
    ):
        """
        Train the model for production.
        """
        logger.info(f"Starting training for production model: {self.model_name} with configuration: "
                    f"Batch Size={BATCH_SIZE}, Epochs={EPOCHS}, Learning Rate={LR}")

        prod_handler = ProdHandler(
            model_name=self.model_name,
            dataset_config=self.dataset_config,
            x=x,
            y=y,
            idx_num=idx_num,
            idx_cat=idx_cat,
            idx_periodic=idx_periodic,
            idx_non_periodic=idx_non_periodic,
            batch_size=BATCH_SIZE,
            num_epochs=EPOCHS,
            learning_rate=LR,
            log_dir=self.log_dir
        )

        start_time = time.time()
        prod_handler.run()
        elapsed_time = time.time() - start_time
        logger.info(f"Production training completed in {elapsed_time / 60:.2f} minutes.")

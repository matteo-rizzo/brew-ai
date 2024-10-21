import time
from typing import Tuple, List

import pandas as pd

from src.classes.data.DataPreprocessor import DataPreprocessor
from src.classes.evaluation.periodicity.CrossValidator import CrossValidator
from src.classes.utils.Logger import Logger
from src.config import APPLY_PCA, NUM_FOLDS, EPOCHS, LR, BATCH_SIZE
from src.functions.utils import detect_periodicity_acf

logger = Logger()


class ExperimentHandler:
    def __init__(self, model_name: str, x: pd.DataFrame, y: pd.Series, log_dir: str):
        """
        ExperimentHandler class to encapsulate the entire experimental workflow.

        :param model_name: Name of the model being used in the experiment.
        :param x: Feature matrix (Pandas DataFrame).
        :param y: Labels (Pandas Series).
        :param log_dir: Directory for saving logs and results.
        """
        self.model_name = model_name
        self.log_dir = log_dir
        self.x = x
        self.y = y

    def identify_periodic_features(self) -> Tuple[List[int], List[int]]:
        """
        Identify periodic and non-periodic numerical features using ACF-based periodicity detection.

        :return: A tuple with two lists: indices of periodic and non-periodic features.
        """
        logger.info("Detecting periodic and non-periodic features.")
        start_time = time.time()

        x_num = self.x.select_dtypes(include=['float64', 'int64'])
        idx_periodic = []
        idx_non_periodic = []

        for column in x_num.columns:
            series = x_num[column].values
            if detect_periodicity_acf(series):
                idx_periodic.append(x_num.columns.get_loc(column))
            else:
                idx_non_periodic.append(x_num.columns.get_loc(column))

        elapsed_time = time.time() - start_time
        logger.info(f"Periodicity detection completed in {elapsed_time:.2f} seconds.")
        logger.info(
            f"Detected {len(idx_periodic)} periodic features and {len(idx_non_periodic)} non-periodic features.")
        return idx_periodic, idx_non_periodic

    def prepare_data(self) -> Tuple[List[int], List[int]]:
        """
        Prepare data by separating numerical and categorical indices.

        :return: A tuple of two lists: indices of numerical features and categorical features.
        """
        logger.info("Identifying numerical and categorical feature indices.")
        idx_num = [self.x.columns.get_loc(col) for col in self.x.select_dtypes(include=['float64', 'int64']).columns]
        idx_cat = [self.x.columns.get_loc(col) for col in self.x.select_dtypes(include=['object', 'category']).columns]

        logger.info(f"Identified {len(idx_num)} numerical features and {len(idx_cat)} categorical features.")
        return idx_num, idx_cat

    def preprocess_data(self):
        """
        Apply preprocessing to the data, such as scaling and optional PCA.
        """
        logger.info(f"Starting data preprocessing. PCA applied: {APPLY_PCA}")
        start_time = time.time()

        data_preprocessor = DataPreprocessor(self.x, self.y, apply_pca=APPLY_PCA)
        preprocessor = data_preprocessor.preprocess()
        self.x = preprocessor.fit_transform(self.x)

        elapsed_time = time.time() - start_time
        logger.info(f"Data preprocessing completed in {elapsed_time:.2f} seconds.")

    def run_cross_validation(self, idx_periodic: List[int], idx_non_periodic: List[int], idx_num: List[int],
                             idx_cat: List[int]):
        """
        Set up and run cross-validation on the model.

        :param idx_periodic: Indices of periodic numerical features.
        :param idx_non_periodic: Indices of non-periodic numerical features.
        :param idx_num: Indices of all numerical features.
        :param idx_cat: Indices of categorical features.
        """
        logger.info(f"Starting cross-validation for model: {self.model_name}")
        logger.info(
            f"Cross-validation setup: Folds={NUM_FOLDS}, Batch Size={BATCH_SIZE}, Epochs={EPOCHS}, Learning Rate={LR}")

        cross_validator = CrossValidator(
            model_name=self.model_name,
            x=self.x,
            y=self.y,
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

    def run_experiment(self):
        """
        Run the full experimental pipeline.
        """
        try:
            logger.info("Starting the evaluation process.")

            # Identify periodic and non-periodic features
            idx_periodic, idx_non_periodic = self.identify_periodic_features()

            # Prepare numerical and categorical feature indices
            idx_num, idx_cat = self.prepare_data()

            # Preprocess data
            self.preprocess_data()

            # Run cross-validation
            self.run_cross_validation(idx_periodic, idx_non_periodic, idx_num, idx_cat)

            logger.info("Evaluation process completed successfully.")
        except Exception as e:
            logger.error(f"An error occurred during evaluation: {str(e)}")

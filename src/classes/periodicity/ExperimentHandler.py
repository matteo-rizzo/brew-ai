import time
from typing import Tuple, List

import pandas as pd

from src.classes.data.DataPreprocessor import DataPreprocessor
from src.classes.data.DatasetLoader import DatasetLoader
from src.classes.periodicity.CrossValidator import CrossValidator
from src.classes.periodicity.prod.ProdHandler import ProdHandler
from src.classes.utils.Logger import Logger
from src.classes.utils.PeriodicityDetector import PeriodicityDetector
from src.config import NUM_FOLDS, EPOCHS, LR, BATCH_SIZE, CLASSIFICATION

logger = Logger()


class ExperimentHandler:
    def __init__(self, model_name: str, dataset_id: str, x: pd.DataFrame, y: pd.Series, log_dir: str, mode: str = "cv"):
        """
        ExperimentHandler class to encapsulate the entire experimental workflow.

        :param model_name: Name of the model being used in the experiment.
        :param x: Feature matrix (Pandas DataFrame).
        :param y: Labels (Pandas Series).
        :param log_dir: Directory for saving logs and results.
        :param mode: Type of experiment, "cv" or "prod".
        """
        self.model_name = model_name
        self.log_dir = log_dir
        self.mode = mode
        self.x = x
        self.y = y
        logger.info(f"Experiment initialized with model: {model_name}, and dataset: {dataset_id} "
                    f"Feature matrix shape: {x.shape}, Labels shape: {y.shape}")

        self.dataset_config = DatasetLoader().get_dataset_config(dataset_id)

        # Precompute numerical and categorical columns for reuse
        self.cat_cols = self.dataset_config["cat_cols"]
        self.feature_selection = self.dataset_config.get("feature_selection", self.x.columns)
        self.numerical_columns = [col for col in self.x.columns if
                                  (col not in self.cat_cols) and (col in self.feature_selection)]
        self.categorical_columns = [col for col in self.x.columns if
                                    (col in self.cat_cols) and (col in self.feature_selection)]

    def reorder_columns(self):
        """
        Reorder columns so that numerical columns appear before categorical columns.
        """
        ordered_columns = self.numerical_columns + self.categorical_columns
        logger.info(f"Reordering columns: {len(self.numerical_columns)} numerical, "
                    f"{len(self.categorical_columns)} categorical")
        self.x = self.x[ordered_columns]

    def identify_periodic_features(self) -> Tuple[List[int], List[int]]:
        """
        Identify periodic and non-periodic numerical features using ACF-based periodicity detection.

        :return: A tuple with two lists: indices of periodic and non-periodic features.
        """
        logger.info("Starting detection of periodic and non-periodic numerical features.")
        start_time = time.time()

        x_num_cols = [col for col in self.numerical_columns if col != 'month']
        idx_periodic, idx_non_periodic = [], []

        logger.info(f"Analyzing periodicity for {len(x_num_cols)} numerical features.")
        for column in x_num_cols:
            series = self.x[column].values
            if PeriodicityDetector().detect_periodicity_acf(series):
                logger.debug(f"Feature '{column}' detected as periodic.")
                idx_periodic.append(self.x.columns.get_loc(column))
            else:
                logger.debug(f"Feature '{column}' detected as non-periodic.")
                idx_non_periodic.append(self.x.columns.get_loc(column))

        elapsed_time = time.time() - start_time
        logger.info(f"Periodicity detection completed in {elapsed_time:.2f} seconds. "
                    f"Detected {len(idx_periodic)} periodic features and {len(idx_non_periodic)} non-periodic features.")
        return idx_periodic, idx_non_periodic

    def prepare_data(self) -> Tuple[List[int], List[int]]:
        """
        Prepare data by separating numerical and categorical indices after reordering the columns.

        :return: A tuple of two lists: indices of numerical features and categorical features.
        """
        start_time = time.time()

        idx_num = [self.x.columns.get_loc(col) for col in self.numerical_columns]
        idx_cat = [self.x.columns.get_loc(col) for col in self.categorical_columns]

        elapsed_time = time.time() - start_time
        logger.info(f"Data preparation completed in {elapsed_time:.2f} seconds. "
                    f"Identified {len(idx_num)} numerical features and {len(idx_cat)} categorical features.")
        return idx_num, idx_cat

    def preprocess_data(self):
        """
        Apply preprocessing to the data, such as scaling and optional PCA.
        """
        logger.info("Starting data preprocessing.")
        start_time = time.time()

        preprocessor = DataPreprocessor(self.x, self.y, self.categorical_columns, self.numerical_columns)
        x_original_shape = self.x.shape
        self.x = preprocessor.make_preprocessor().fit_transform(self.x)
        if CLASSIFICATION:
            self.y = preprocessor.encode_target()

        elapsed_time = time.time() - start_time
        logger.info(f"Data preprocessing completed in {elapsed_time:.2f} seconds. "
                    f"Data shape changed from {x_original_shape} to {self.x.shape}.")

    def run_cross_validation(
            self,
            idx_periodic: List[int],
            idx_non_periodic: List[int],
            idx_num: List[int],
            idx_cat: List[int]
    ):
        """
        Set up and run cross-validation on the model.

        :param idx_periodic: Indices of periodic numerical features.
        :param idx_non_periodic: Indices of non-periodic numerical features.
        :param idx_num: Indices of all numerical features.
        :param idx_cat: Indices of categorical features.
        """
        logger.info(f"Starting cross-validation for model: {self.model_name} with configuration: "
                    f"Folds={NUM_FOLDS}, Batch Size={BATCH_SIZE}, Epochs={EPOCHS}, Learning Rate={LR}")

        cross_validator = CrossValidator(
            model_name=self.model_name,
            dataset_config=self.dataset_config,
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

    def run_production(
            self,
            idx_periodic: List[int],
            idx_non_periodic: List[int],
            idx_num: List[int],
            idx_cat: List[int]
    ):
        """
        Train the model for production.

        :param idx_periodic: Indices of periodic numerical features.
        :param idx_non_periodic: Indices of non-periodic numerical features.
        :param idx_num: Indices of all numerical features.
        :param idx_cat: Indices of categorical features.
        """
        logger.info(f"Starting training for production model: {self.model_name} with configuration: "
                    f"Batch Size={BATCH_SIZE}, Epochs={EPOCHS}, Learning Rate={LR}")

        prod_handler = ProdHandler(
            model_name=self.model_name,
            dataset_config=self.dataset_config,
            x=self.x,
            y=self.y,
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

    def run_experiment(self):
        """
        Run the full experimental pipeline.
        """
        try:
            logger.info("Starting the experiment.")

            # Reorder columns first
            self.reorder_columns()

            # Identify periodic and non-periodic features
            idx_periodic, idx_non_periodic = self.identify_periodic_features()

            # Prepare numerical and categorical feature indices
            idx_num, idx_cat = self.prepare_data()

            # Preprocess data
            self.preprocess_data()

            if self.mode == "cv":
                # Run cross-validation
                self.run_cross_validation(idx_periodic, idx_non_periodic, idx_num, idx_cat)
            elif self.mode == "prod":
                # Train model for production
                self.run_production(idx_periodic, idx_non_periodic, idx_num, idx_cat)

            logger.info("Experiment completed successfully.")
        except Exception as e:
            logger.error(f"An error occurred during evaluation: {str(e)}")

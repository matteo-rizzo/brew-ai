from typing import Dict

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.classes.grid_search.CrossValidator import CrossValidator
from src.classes.grid_search.ModelConfigFactory import ModelConfigFactory
from src.classes.utils.Logger import Logger
from src.functions.utils import make_model_subdirectory
from src.config import CV_METRICS, NUM_FOLDS

logger = Logger()


class ExperimentHandler:
    """
    Handles model training, tuning, and cross-validation for different models.
    """

    def __init__(self, x: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer, log_dir: str) -> None:
        """
        Initialize the ExperimentHandler with data, preprocessor, and logging directory.

        :param x: Features for modeling (Pandas DataFrame).
        :param y: Target variable for prediction (Pandas Series).
        :param preprocessor: Preprocessing pipeline for the data.
        :param log_dir: Directory to save the logs and best parameters.
        """
        self.x, self.y = x, y
        self.preprocessor = preprocessor
        self.log_dir = log_dir
        self.model_factory = ModelConfigFactory()

        logger.info("ExperimentHandler initialized with the provided dataset and preprocessor.")

    def create_pipeline_and_grid_search(self, model: BaseEstimator, param_grid: Dict) -> GridSearchCV:
        """
        Create a pipeline with preprocessing and the model, then initialize GridSearchCV.

        :param model: Machine learning model.
        :param param_grid: Hyperparameter grid for the model.
        :return: Initialized GridSearchCV object.
        """
        pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('model', model)])
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=NUM_FOLDS,
            scoring=CV_METRICS,
            n_jobs=-1,
            refit="r2",
            return_train_score=False,
            verbose=3
        )
        return grid_search

    def run_experiment(self) -> None:
        """
        Train and evaluate models using cross-validation and hyperparameter tuning.
        """
        logger.info("Starting the model training and evaluation process...")

        model_configs = self.model_factory.get_model_configurations()

        for model_name, (model, param_grid) in model_configs.items():
            try:
                logger.info(f"Training and evaluating model: [bold]{model_name}[/bold]")

                # Create model-specific log directory
                model_log_dir = self._prepare_model_directory(model_name)

                # Create the GridSearchCV object with pipeline and param grid
                grid_search = self.create_pipeline_and_grid_search(model, param_grid)

                # Perform cross-validation for the model
                self._perform_cross_validation(grid_search, model_name, model_log_dir)
            except Exception as e:
                logger.error(f"Error during training of [bold]{model_name}[/bold]: {str(e)}")

    def _prepare_model_directory(self, model_name: str) -> str:
        """
        Create a subdirectory for storing results of the given model.

        :param model_name: The name of the model being trained.
        :return: Path to the model's log directory.
        """
        model_log_dir = make_model_subdirectory(model_name, self.log_dir)
        logger.info(f"Created model-specific directory at {model_log_dir}")
        return model_log_dir

    def _perform_cross_validation(self, grid_search: GridSearchCV, model_name: str, model_log_dir: str) -> None:
        """
        Perform cross-validation for the given model and retrieve the best estimator.

        :param grid_search: GridSearchCV object for hyperparameter tuning.
        :param model_name: Name of the model being trained.
        :param model_log_dir: Subdirectory for storing cross-validation results.
        """
        logger.info(f"Performing cross-validation for [bold]{model_name}[/bold]...")

        # Perform cross-validation using CrossValidator
        CrossValidator(grid_search).perform_cross_validation(
            model_name=model_name,
            x=self.x,
            y=self.y,
            log_dir=model_log_dir
        )

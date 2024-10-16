import json
import os
from typing import Dict

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.classes.evaluation.CrossValidator import CrossValidator
from src.classes.evaluation.MetricsCalculator import MetricsCalculator
from src.classes.evaluation.VisualizationsManager import VisualizationsManager
from src.classes.utils.Logger import Logger
from src.classes.evaluation.ModelConfigFactory import ModelConfigFactory
from src.functions.utils import make_model_subdirectory
from src.settings import TEST_SIZE, RANDOM_SEED

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
        :param log_dir: Directory to save the plots and best parameters.
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
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
        return GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, return_train_score=True)

    def run_experiment(self) -> None:
        """
        Train and evaluate models using cross-validation and hyperparameter tuning.
        """
        logger.info("Starting the model training and evaluation process...")

        for model_name, (model, param_grid) in self.model_factory.get_model_configurations().items():
            try:
                logger.info(f"Training and evaluating model: {model_name}...")

                # Step 1: Create subdirectory for each model
                model_log_dir = self._prepare_model_directory(model_name)

                # Step 2: Create pipeline and perform grid search
                grid_search = self.create_pipeline_and_grid_search(model, param_grid)

                # Step 3: Perform cross-validation
                best_estimator = self._perform_cross_validation(grid_search, model_name, model_log_dir)

                # Step 4: Evaluate the best model and visualize results
                self._evaluate_and_visualize_model(best_estimator, model_name, model_log_dir)

                # Step 5: Save best model parameters
                self._save_best_parameters(grid_search, model_name, model_log_dir)

            except Exception as e:
                logger.error(f"Error during training of {model_name}: {str(e)}")

    def _prepare_model_directory(self, model_name: str) -> str:
        """
        Create a subdirectory for storing results of the given model.

        :param model_name: The name of the model being trained.
        :return: Path to the model's log directory.
        """
        model_log_dir = make_model_subdirectory(model_name, self.log_dir)
        logger.info(f"Created model-specific directory at {model_log_dir}")
        return model_log_dir

    def _perform_cross_validation(self, grid_search: GridSearchCV, model_name: str,
                                  model_log_dir: str) -> BaseEstimator:
        """
        Perform cross-validation for the given model and retrieve the best estimator.

        :param grid_search: GridSearchCV object for hyperparameter tuning.
        :param model_name: Name of the model being trained.
        :param model_log_dir: Subdirectory for storing cross-validation results.
        :return: Best estimator from GridSearchCV.
        """
        logger.info(f"Performing cross-validation for {model_name}...")

        # Perform cross-validation and return the best model
        return CrossValidator.perform_cross_validation(
            grid_search=grid_search,
            model_name=model_name,
            x=self.x_train,
            y=self.y_train,
            log_dir=model_log_dir
        ).best_estimator_

    def _evaluate_and_visualize_model(self, best_model: BaseEstimator, model_name: str, model_log_dir: str) -> None:
        """
        Evaluate the model on a hold-out test set and generate visual analysis.

        :param best_model: Best estimator after cross-validation.
        :param model_name: Name of the model being trained.
        :param model_log_dir: Subdirectory for storing evaluation results.
        """
        logger.info(f"Evaluating and visualizing model: {model_name}...")

        # Predict on test set
        y_pred = best_model.predict(self.x_test)

        # Step 1: Evaluate the model and save metrics
        MetricsCalculator.evaluate_model(model_name, self.y_test, y_pred, model_log_dir)

        # Step 2: Generate and save visualizations
        VisualizationsManager.visualize_model_results(best_model, model_name, self.y_test, y_pred, model_log_dir)

    @staticmethod
    def _save_best_parameters(grid_search: GridSearchCV, model_name: str, model_log_dir: str) -> None:
        """
        Save the best parameters for the model to a JSON file in the model-specific subdirectory.

        :param grid_search: The GridSearchCV object after training.
        :param model_name: The name of the model whose parameters are being saved.
        :param model_log_dir: Subdirectory for storing the best parameters.
        """
        try:
            best_params = grid_search.best_params_
            logger.info(f"Best parameters for {model_name}: {best_params}")

            # Define the file path to save the parameters as JSON
            params_file_path = os.path.join(model_log_dir, f"{model_name}_best_params.json")

            # Save the best parameters to a JSON file
            with open(params_file_path, 'w') as file:
                json.dump(best_params, file, indent=4)

            logger.info(f"Best parameters saved to {params_file_path}")

        except Exception as e:
            logger.error(f"Failed to save best parameters for {model_name}: {e}")

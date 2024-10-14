from typing import Dict, Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split

from src.classes.CrossValidator import CrossValidator
from src.classes.Logger import Logger
from src.classes.ModelEvaluator import ModelEvaluator
from src.classes.PipelineBuilder import PipelineBuilder

logger = Logger()


class ModelHandler:
    """
    Handles model training, tuning, and cross-validation for different models.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, preprocessor: ColumnTransformer, logdir: str) -> None:
        """
        Initialize the ModelHandler with data, preprocessor, and logging directory.

        :param x: Features for modeling (NumPy array).
        :param y: Target variable for prediction (NumPy array).
        :param preprocessor: Preprocessing pipeline for the data.
        :param logdir: Directory to save the plots.
        """
        self.x = x
        self.y = y
        self.preprocessor = preprocessor
        self.logdir = logdir
        self.models_and_params = PipelineBuilder.get_model_parameters()
        self.cv_results: Dict[str, Dict[str, Any]] = {}
        self.final_results: Dict[str, Dict[str, float]] = {}
        logger.info("ModelHandler initialized with the provided dataset and preprocessor.")

    def train_and_evaluate(self) -> None:
        """
        Train and evaluate models using cross-validation and hyperparameter tuning.
        """
        logger.info("Starting the model training and evaluation process...")

        for model_name, (model, param_grid) in self.models_and_params.items():
            try:
                logger.info(f"Training and evaluating [bold blue]{model_name}[/bold blue]...")

                # Step 1: Create pipeline and grid search
                grid_search = PipelineBuilder.create_pipeline_and_grid_search(self.preprocessor, model, param_grid)

                # Step 2: Perform cross-validation
                self._perform_cross_validation(grid_search, model_name)

                # Step 3: Train and evaluate the model
                self._train_and_evaluate_model(grid_search, model_name)

            except Exception as e:
                logger.error(f"Error occurred during training of {model_name}: {str(e)}")

    def _perform_cross_validation(self, grid_search: GridSearchCV, model_name: str) -> None:
        """
        Perform cross-validation for the given model and store the results.

        :param grid_search: GridSearchCV object for hyperparameter tuning.
        :param model_name: Name of the model being trained.
        """
        logger.info(f"Performing cross-validation for {model_name}...")
        CrossValidator.perform_cross_validation(
            grid_search=grid_search,
            model_name=model_name,
            x=self.x,
            y=self.y,
            cv_results=self.cv_results
        )

    def _train_and_evaluate_model(self, grid_search: GridSearchCV, model_name: str) -> None:
        """
        Train and evaluate the model on a hold-out test set and generate visual analysis.

        :param grid_search: GridSearchCV object for model fitting and parameter tuning.
        :param model_name: Name of the model being trained.
        """
        logger.info(f"Training {model_name} with the best parameters...")

        # Step 1: Train-test split
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        # Step 2: Fit the model and make predictions
        grid_search.fit(x_train, y_train)
        y_pred = grid_search.predict(x_test)

        # Step 3: Evaluate the model and generate visual analysis
        logger.info(f"Evaluating and visualizing {model_name}...")
        ModelEvaluator.evaluate_and_visualize_model(
            grid_search=grid_search,
            model_name=model_name,
            y_test=y_test,
            y_pred=y_pred,
            logdir=self.logdir,
            final_results=self.final_results
        )

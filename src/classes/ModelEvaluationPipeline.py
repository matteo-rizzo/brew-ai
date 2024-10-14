from typing import Dict

import numpy as np
from sklearn.model_selection import GridSearchCV

from src.classes.Logger import Logger
from src.classes.ModelEvaluator import ModelEvaluator
from src.classes.Visualizer import Visualizer

# Initialize custom logger
logger = Logger()


class ModelEvaluationPipeline:
    """
    Coordinates model evaluation and visualization.
    """

    @staticmethod
    def evaluate_and_visualize_model(grid_search: GridSearchCV, model_name: str, y_test: np.ndarray,
                                     y_pred: np.ndarray, log_dir: str) -> None:
        """
        Evaluate the model and generate visual analysis plots.

        :param grid_search: GridSearchCV object used to find the best estimator
        :param model_name: Name of the model
        :param y_test: True values for the test set
        :param y_pred: Predicted values for the test set
        :param log_dir: Directory to save plots
        """
        try:
            # Evaluate the model
            ModelEvaluator.evaluate_model(model_name, y_test, y_pred, log_dir)

            # Visualize model results
            Visualizer.visualize_model_results(grid_search, model_name, y_test, y_pred, log_dir)

        except Exception as e:
            logger.error(f"Failed to evaluate and visualize model {model_name}: {e}")
            raise

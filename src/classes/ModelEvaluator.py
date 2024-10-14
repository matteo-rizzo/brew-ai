import json
import os
from typing import Dict

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, \
    explained_variance_score

from src.classes.Logger import Logger

# Initialize custom logger
logger = Logger()


class ModelEvaluator:
    """
    Evaluates the model performance by calculating evaluation metrics.
    """

    @staticmethod
    def calculate_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics for the model.

        :param y_test: True target values
        :param y_pred: Predicted target values
        :return: A dictionary containing all calculated metrics
        """
        try:
            metrics = {
                'R^2': r2_score(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'MAPE': mean_absolute_percentage_error(y_test, y_pred),
                'Explained Variance': explained_variance_score(y_test, y_pred)
            }

            logger.info("Metrics calculated successfully.")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise

    @staticmethod
    def evaluate_model(model_name: str, y_test: np.ndarray, y_pred: np.ndarray, log_dir: str) -> None:
        """
        Calculate and store evaluation metrics for each model and save to log_dir as a JSON file.

        :param model_name: Name of the model being evaluated
        :param y_test: True target values
        :param y_pred: Predicted target values
        :param log_dir: Log directory
        """
        try:
            # Calculate metrics
            metrics = ModelEvaluator.calculate_metrics(y_test, y_pred)

            # Log metrics as a table using Logger
            logger.log_metrics_table(model_name, metrics)

            # Save metrics to a JSON file
            ModelEvaluator._save_metrics_to_json(model_name, metrics, log_dir)

        except Exception as e:
            logger.error(f"Failed to evaluate the model {model_name}: {e}")
            raise

    @staticmethod
    def _save_metrics_to_json(model_name: str, metrics: Dict[str, float], log_dir: str) -> None:
        """
        Save the evaluation metrics to a JSON file in the log directory.

        :param model_name: Name of the model being evaluated
        :param metrics: Dictionary containing calculated metrics
        :param log_dir: Directory where the metrics will be saved
        """
        try:
            # Define the file path to save the metrics in JSON format
            metrics_file_path = os.path.join(log_dir, f"{model_name}_metrics.json")

            # Save the metrics to a JSON file
            with open(metrics_file_path, 'w') as json_file:
                json.dump(metrics, json_file, indent=4)

            logger.info(f"Metrics saved as JSON to {metrics_file_path}")

        except Exception as e:
            logger.error(f"Failed to save metrics for {model_name} as JSON: {e}")
            raise

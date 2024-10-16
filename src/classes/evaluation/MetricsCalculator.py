import json
import os
from typing import Dict

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, \
    explained_variance_score

from src.classes.utils.Logger import Logger

# Initialize custom logger
logger = Logger()


class MetricsCalculator:
    """
    Evaluates the model performance by calculating evaluation metrics.
    """

    @staticmethod
    def calculate_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics for the model.

        This method computes several performance metrics to evaluate regression models,
        comparing the true target values (y_test) against the predicted values (y_pred):

        - **R^2 (Coefficient of Determination)**: Measures how well the predicted values
          approximate the real data. It ranges from 0 to 1, where a higher score indicates
          better predictive accuracy. A negative value implies a worse fit than a horizontal line.

        - **MSE (Mean Squared Error)**: The average of the squared differences between
          actual and predicted values. It penalizes large errors more than smaller ones,
          making it sensitive to outliers.

        - **RMSE (Root Mean Squared Error)**: The square root of the mean squared error,
          which provides a measure of how much error to expect in predictions. It's expressed
          in the same units as the target variable, making it easier to interpret.

        - **MAE (Mean Absolute Error)**: The average of the absolute differences between
          actual and predicted values. MAE is less sensitive to outliers than MSE.

        - **MAPE (Mean Absolute Percentage Error)**: The average of the absolute percentage
          differences between actual and predicted values. It provides an indication of
          how large the prediction errors are relative to the actual values, expressed as a percentage.

        - **Explained Variance**: Measures the proportion of the variance in the target
          variable that is predictable from the features. It ranges from 0 to 1, where 1
          indicates perfect predictive accuracy.

        :param y_test: True target values (NumPy array).
        :param y_pred: Predicted target values (NumPy array).
        :return: A dictionary containing all calculated metrics.
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
            metrics = MetricsCalculator.calculate_metrics(y_test, y_pred)

            # Log metrics as a table using Logger
            logger.log_metrics_table(model_name, metrics)

            # Save metrics to a JSON file
            MetricsCalculator._save_metrics_to_json(model_name, metrics, log_dir)

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

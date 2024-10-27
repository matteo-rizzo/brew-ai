import json
import os
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from src.classes.utils.Logger import Logger
from src.config import CLASSIFICATION

# Initialize custom logger
logger = Logger()


class MetricsCalculator:
    """
    Evaluates the model performance by calculating evaluation metrics.
    """

    @staticmethod
    def _compute_classification_metrics(y_test: np.ndarray, y_pred_logits: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics for a classification model by converting logits to predictions.

        This method computes several performance metrics to evaluate classification models,
        comparing the true target values (y_test) against the predicted values (y_pred).

        - **Accuracy**: The ratio of correctly predicted samples to the total number of samples.
        - **Precision**: The ratio of true positives to the sum of true and false positives.
        - **Recall**: The ratio of true positives to the sum of true positives and false negatives.
        - **F1 Score**: The harmonic mean of precision and recall, balancing the two metrics.

        :param y_test: True target values (NumPy array).
        :param y_pred_logits: Logit predictions (NumPy array).
        :return: A dictionary containing all calculated metrics.
        """
        try:
            # Convert logits to predictions
            if y_pred_logits.shape[1] == 1:
                # Binary classification: apply a threshold of 0 for logits
                y_pred = (y_pred_logits > 0).astype(int).squeeze()
            else:
                # Multi-class classification: use argmax to get class predictions
                y_pred = np.argmax(y_pred_logits, axis=1)

            # Calculate classification metrics
            metrics = {
                'Accuracy': float(accuracy_score(y_test, y_pred)),
                'Precision': float(precision_score(y_test, y_pred, average='binary')),
                'Recall': float(recall_score(y_test, y_pred, average='binary')),
                'F1 Score': float(f1_score(y_test, y_pred, average='binary'))
            }

            logger.info("Classification metrics calculated successfully.")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")
            raise

    @staticmethod
    def _compute_regression_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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

        :param y_test: True target values (NumPy array).
        :param y_pred: Predicted target values (NumPy array).
        :return: A dictionary containing all calculated metrics.
        """
        try:
            metrics = {
                'R^2': float(r2_score(y_test, y_pred)),
                'MSE': float(mean_squared_error(y_test, y_pred)),
                'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'MAE': float(mean_absolute_error(y_test, y_pred)),
                'MAPE': float(mean_absolute_percentage_error(y_test, y_pred))
            }

            logger.info("Metrics calculated successfully.")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise

    @staticmethod
    def calculate_metrics(
            model_name: str,
            y_test: np.ndarray,
            y_pred: np.ndarray,
            log_dir: str,
            fold: int = None,
            classification: bool = CLASSIFICATION
    ) -> Dict:
        """
        Calculate and store evaluation metrics for each model and save to log_dir as a JSON file.

        :param model_name: Name of the model being evaluated
        :param y_test: True target values
        :param y_pred: Predicted target values
        :param log_dir: Log directory
        :param fold: Current fold for logging
        :param classification: Flag to compute classification vs regression metrics
        """
        try:
            # Calculate metrics
            if classification:
                metrics = MetricsCalculator._compute_classification_metrics(y_test, y_pred)
            else:
                metrics = MetricsCalculator._compute_regression_metrics(y_test, y_pred)

            print(metrics)

            # Log metrics as a table using Logger
            logger.log_metrics_table(model_name, metrics)

            # Save metrics to a JSON file
            MetricsCalculator.save_metrics_to_json(model_name, metrics, log_dir, fold)

            return metrics
        except Exception as e:
            logger.error(f"Failed to evaluate the model {model_name}: {e}")
            raise

    @staticmethod
    def save_metrics_to_json(model_name: str, metrics: Dict[str, float], log_dir: str, fold: int = None) -> None:
        """
        Save the evaluation metrics to a JSON file in the log directory.

        :param model_name: Name of the model being evaluated
        :param metrics: Dictionary containing calculated metrics
        :param log_dir: Directory where the metrics will be saved
        :param fold: Current fold for logging
        """
        fold = "" if fold is None else fold
        try:
            # Define the file path to save the metrics in JSON format
            metrics_file_path = os.path.join(log_dir, f"{model_name}{fold}_metrics.json")

            # Save the metrics to a JSON file
            with open(metrics_file_path, 'w') as json_file:
                json.dump(metrics, json_file, indent=4)

            logger.info(f"Metrics saved as JSON to {metrics_file_path}")

        except Exception as e:
            logger.error(f"Failed to save metrics for {model_name} as JSON: {e}")
            raise

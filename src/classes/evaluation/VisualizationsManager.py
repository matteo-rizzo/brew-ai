import os
from typing import Any, Dict

import numpy as np
from sklearn.base import BaseEstimator

from src.classes.utils.Logger import Logger
from src.classes.utils.Plotter import Plotter

# Initialize custom logger
logger = Logger()


class VisualizationsManager:
    """
    Manages the generation of visual analysis plots for model results.
    """

    @staticmethod
    def visualize_model_results(estimator: BaseEstimator, model_name: str, y_test: np.ndarray, y_pred: np.ndarray,
                                log_dir: str) -> None:
        """
        Generate and save visual analysis plots.

        :param estimator: Best estimator (fitted pipeline with model and preprocessor)
        :param model_name: Name of the model
        :param y_test: True values for the test set
        :param y_pred: Predicted values for the test set
        :param log_dir: Directory to save the generated plots
        """
        logger.info(f"Generating visual analysis for {model_name}...")

        # Paths for saving the plots
        actual_vs_pred_path = os.path.join(log_dir, f"{model_name}_actual_vs_pred.png")
        residuals_path = os.path.join(log_dir, f"{model_name}_residuals.png")
        feature_importance_path = os.path.join(log_dir, f"{model_name}_feature_importance.png")

        # Plot actual vs predicted values
        VisualizationsManager._plot_actual_vs_predicted(y_test, y_pred, actual_vs_pred_path)

        # Plot residuals
        VisualizationsManager._plot_residuals(y_test, y_pred, residuals_path)

        # Extract preprocessor and model steps from the estimator pipeline
        preprocessor = estimator.named_steps['preprocessor']
        model = estimator.named_steps['model']

        # Check if feature importance is available and plot it
        if hasattr(model, 'coef_') or hasattr(model, 'feature_importances_'):
            VisualizationsManager._plot_feature_importance(model, preprocessor, model_name, feature_importance_path)
        else:
            logger.warning(f"Feature importance is not supported for {model_name}.")

    @staticmethod
    def _plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
        """Plot actual vs predicted values."""
        Plotter.plot_actual_vs_predicted(y_test, y_pred, save_path)
        logger.info(f"Actual vs Predicted plot saved at {save_path}")

    @staticmethod
    def _plot_residuals(y_test: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
        """Plot residuals distribution."""
        Plotter.plot_residuals(y_test, y_pred, save_path)
        logger.info(f"Residuals plot saved at {save_path}")

    @staticmethod
    def _plot_feature_importance(model: BaseEstimator, preprocessor: Any, model_name: str, save_path: str) -> None:
        """
        Plot and save feature importance for models that support it.

        :param model: Trained model with feature importance or coefficients
        :param preprocessor: Preprocessing pipeline used in the model
        :param model_name: Name of the model
        :param save_path: Path to save the feature importance plot
        """
        try:
            # Extract feature names
            feature_names = VisualizationsManager.get_feature_names(preprocessor)
            # Plot feature importance
            Plotter.plot_feature_importance(model, feature_names, save_path)
            logger.info(f"Feature importance plot saved at {save_path}")
        except ValueError as ve:
            logger.error(f"Error generating feature importance for {model_name}: {ve}")
        except Exception as e:
            logger.error(f"Unexpected error generating feature importance for {model_name}: {e}")

    @staticmethod
    def get_feature_names(preprocessor: Any) -> Dict:
        """
        Extract feature names from the preprocessor after fitting.

        :param preprocessor: The fitted preprocessor object
        :return: A dictionary containing numerical and categorical feature names
        """
        try:
            # Extract numerical feature names
            num_transformer = preprocessor.named_transformers_['num']
            if 'pca' in num_transformer.named_steps:
                n_components = num_transformer.named_steps['pca'].n_components_
                num_feature_names = [f"PC{i + 1}" for i in range(n_components)]
            else:
                num_feature_names = num_transformer.named_steps['scaler'].get_feature_names_out()

            # Extract categorical feature names
            cat_transformer = preprocessor.named_transformers_['cat']
            cat_feature_names = cat_transformer.get_feature_names_out()

            return {"num_cols": num_feature_names, "cat_cols": cat_feature_names}

        except Exception as e:
            logger.error(f"Error extracting feature names from preprocessor: {e}")
            raise

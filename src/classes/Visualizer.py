import os
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import GridSearchCV

from src.classes.Logger import Logger
from src.classes.PlotHandler import PlotHandler

# Initialize custom logger
logger = Logger()


class Visualizer:
    """
    Handles the generation of visual analysis plots for model results.
    """

    @staticmethod
    def visualize_model_results(grid_search: GridSearchCV, model_name: str, y_test: np.ndarray, y_pred: np.ndarray,
                                log_dir: str) -> None:
        """
        Generate visual analysis plots.

        :param grid_search: GridSearchCV object used to find the best estimator
        :param model_name: Name of the model
        :param y_test: True values for the test set
        :param y_pred: Predicted values for the test set
        :param log_dir: Directory to save plots
        """
        logger.info(f"Generating visual analysis for [bold blue]{model_name}[/bold blue]...")

        # Paths for saving the plots
        actual_vs_pred_path = os.path.join(log_dir, f"{model_name}_actual_vs_pred.png")
        residuals_path = os.path.join(log_dir, f"{model_name}_residuals.png")
        feature_importance_path = os.path.join(log_dir, f"{model_name}_feature_importance.png")

        # Plot actual vs predicted values
        PlotHandler.plot_actual_vs_predicted(y_test, y_pred, actual_vs_pred_path)
        logger.info(f"Actual vs Predicted plot saved at {actual_vs_pred_path}")

        # Plot residuals
        PlotHandler.plot_residuals(y_test, y_pred, residuals_path)
        logger.info(f"Residuals plot saved at {residuals_path}")

        # Extract model and preprocessor for feature importance
        if hasattr(grid_search, 'best_estimator_'):
            estimator = grid_search.best_estimator_
        else:
            estimator = grid_search  # grid_search might be the model pipeline

        preprocessor = estimator.named_steps['preprocessor']
        model = estimator.named_steps['model']

        if hasattr(model, 'coef_') or hasattr(model, 'feature_importances_'):
            try:
                # Extract numerical and categorical feature names
                feature_names = Visualizer.get_feature_names(preprocessor)

                # Plot feature importance
                PlotHandler.plot_feature_importance(model, feature_names, feature_importance_path)
                logger.info(f"Feature importance plot saved at {feature_importance_path}")

            except ValueError as ve:
                logger.error(f"Error generating feature importance for {model_name}: {ve}")
            except Exception as e:
                logger.error(f"Unexpected error generating feature importance for {model_name}: {e}")
        else:
            logger.warning(f"[bold yellow]Feature importance is not supported for {model_name}[/bold yellow].")

    @staticmethod
    def get_feature_names(preprocessor: Any) -> Dict:
        """
        Extract feature names from the preprocessor after fitting.

        :param preprocessor: The fitted preprocessor object
        :return: A dictionary of feature names
        """
        try:
            num_transformer = preprocessor.named_transformers_['num']
            if 'pca' in num_transformer.named_steps:
                n_components = num_transformer.named_steps['pca'].n_components_
                num_feature_names = [f"PC{i + 1}" for i in range(n_components)]
            else:
                num_feature_names = num_transformer.named_steps['scaler'].get_feature_names_out()

            cat_transformer = preprocessor.named_transformers_['cat']
            cat_feature_names = cat_transformer.get_feature_names_out()

            return {"num_cols": num_feature_names, "cat_cols": cat_feature_names}

        except Exception as e:
            logger.error(f"Error extracting feature names from preprocessor: {e}")
            raise

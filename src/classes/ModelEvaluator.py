import os
from typing import Dict, Any

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, \
    explained_variance_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.classes.Logger import Logger
from src.classes.PlotHandler import PlotHandler

# Initialize custom logger
logger = Logger()


class ModelEvaluator:
    """
    Evaluates the model performance and generates visual analysis.
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
            logger.info(f"Metrics calculated successfully: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise

    @staticmethod
    def evaluate_model(model_name: str, y_test: np.ndarray, y_pred: np.ndarray,
                       final_results: Dict[str, Dict[str, float]]) -> None:
        """
        Calculate and store evaluation metrics for each model.

        :param model_name: Name of the model being evaluated
        :param y_test: True target values
        :param y_pred: Predicted target values
        :param final_results: Dictionary to store the final evaluation results
        """
        try:
            metrics = ModelEvaluator.calculate_metrics(y_test, y_pred)
            logger.info(f"Evaluation metrics for [bold blue]{model_name}[/bold blue]: {metrics}")
            final_results[model_name] = metrics
        except Exception as e:
            logger.error(f"Failed to evaluate the model {model_name}: {e}")
            raise

    @staticmethod
    def visualize_model_results(grid_search: GridSearchCV, model_name: str, y_test: np.ndarray, y_pred: np.ndarray,
                                logdir: str) -> None:
        """
        Generate visual analysis plots.

        :param grid_search: GridSearchCV object used to find the best estimator
        :param model_name: Name of the model
        :param y_test: True values for the test set
        :param y_pred: Predicted values for the test set
        :param logdir: Directory to save plots
        """
        logger.info(f"Generating visual analysis for [bold blue]{model_name}[/bold blue]...")

        # Construct paths for saving the plots
        actual_vs_pred_path = os.path.join(logdir, f"{model_name}_actual_vs_pred.png")
        residuals_path = os.path.join(logdir, f"{model_name}_residuals.png")
        feature_importance_path = os.path.join(logdir, f"{model_name}_feature_importance.png")

        # Plot actual vs predicted values
        PlotHandler.plot_actual_vs_predicted(y_test, y_pred, actual_vs_pred_path)
        logger.info(f"Actual vs Predicted plot saved at {actual_vs_pred_path}")

        # Plot residuals
        PlotHandler.plot_residuals(y_test, y_pred, residuals_path)
        logger.info(f"Residuals plot saved at {residuals_path}")

        # Determine the estimator and preprocessor
        if hasattr(grid_search, 'best_estimator_'):
            estimator = grid_search.best_estimator_
            print("hello")
        else:
            estimator = grid_search  # grid_search is actually the pipeline or model

        # Generate feature importance plot if applicable
        model = estimator.named_steps['model']
        preprocessor = estimator.named_steps['preprocessor']

        if hasattr(model, 'coef_'):
            try:
                # Extract numerical and categorical feature names
                feature_names = ModelEvaluator.get_feature_names(preprocessor)

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
        :return: An array of feature names
        """
        feature_names = []

        try:
            # Numerical features
            num_transformer = preprocessor.named_transformers_['num']
            if isinstance(num_transformer, Pipeline):
                # Check if PCA is applied
                if 'pca' in num_transformer.named_steps:
                    pca = num_transformer.named_steps['pca']
                    n_components = pca.n_components_
                    num_feature_names = np.array([f'PC{i + 1}' for i in range(n_components)])
                    logger.info(f"PCA applied. Numerical feature names: {num_feature_names}")
                else:
                    scaler = num_transformer.named_steps['scaler']
                    num_feature_names = scaler.get_feature_names_out()
                    logger.info(f"Numerical feature names after scaling: {num_feature_names}")
            else:
                num_feature_names = np.array(preprocessor.transformers_[0][2])  # Original numerical column names

            # Categorical features
            cat_transformer = preprocessor.named_transformers_['cat']
            cat_feature_names = cat_transformer.get_feature_names_out()
            logger.info(f"Categorical feature names after encoding: {cat_feature_names}")

            # Combine all feature names
            return {
                "num_cols": num_feature_names,
                "cat_cols": cat_feature_names
            }
        except Exception as e:
            logger.error(f"Error extracting feature names from preprocessor: {e}")
            raise

    @staticmethod
    def evaluate_and_visualize_model(grid_search: GridSearchCV, model_name: str, y_test: np.ndarray,
                                     y_pred: np.ndarray, logdir: str,
                                     final_results: Dict[str, Dict[str, float]]) -> None:
        """
        Evaluate the model and generate visual analysis plots.

        :param final_results: Dictionary to store evaluation results
        :param grid_search: GridSearchCV object used to find the best estimator
        :param model_name: Name of the model
        :param y_test: True values for the test set
        :param y_pred: Predicted values for the test set
        :param logdir: Directory to save plots
        """
        try:
            # Evaluate the model
            ModelEvaluator.evaluate_model(model_name, y_test, y_pred, final_results)

            # Visualize model results
            ModelEvaluator.visualize_model_results(grid_search, model_name, y_test, y_pred, logdir)

        except Exception as e:
            logger.error(f"Failed to evaluate and visualize model {model_name}: {e}")
            raise

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class PlotHandler:
    """
    Handles the creation and saving of various plots such as actual vs predicted values, residuals, and feature importance.
    """

    @staticmethod
    def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray, save_path: str):
        """Generate scatter plot for actual vs predicted values and save to the given path."""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue', s=60, edgecolor='black', alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
        plt.title('Actual vs Predicted Values', fontsize=16, pad=12)
        plt.xlabel('Actual Tempo di riduzione diacetile', fontsize=14)
        plt.ylabel('Predicted Tempo di riduzione diacetile', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_residuals(y_test: np.ndarray, y_pred: np.ndarray, save_path: str):
        """Generate residuals plot and save to the given path."""
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True, color='green')
        plt.title('Residuals Distribution', fontsize=16, pad=12)
        plt.xlabel('Residuals', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_feature_importance(model, feature_names: Dict, save_path: str):
        """Generate feature importance plot and save to the given path."""
        # Extract the model from the pipeline

        if hasattr(model, 'coef_'):
            # For linear models
            PlotHandler._plot_linear_model_feature_importance(model, feature_names, save_path)
        elif hasattr(model, 'feature_importances_'):
            # For tree-based models like RandomForest or GradientBoosting
            PlotHandler._plot_tree_model_feature_importance(model, feature_names, save_path)
        else:
            print(f"Model {type(model).__name__} does not support feature importance plotting.")

    @staticmethod
    def _plot_linear_model_feature_importance(model, feature_names: Dict, save_path: str):
        """Helper to plot feature importance for linear models."""
        # Extract feature names
        feature_names_num = feature_names["num_cols"]
        feature_names_cat = feature_names["cat_cols"]
        all_feature_names = np.concatenate([feature_names_num, feature_names_cat])

        # Get model coefficients
        coefficients = model.coef_

        # Create DataFrame for feature importance
        feature_importance = pd.DataFrame({
            'Feature': all_feature_names,
            'Coefficient': coefficients,
            'Absolute Coefficient': np.abs(coefficients)
        })

        # Sort by absolute coefficient
        feature_importance = feature_importance.sort_values(by='Absolute Coefficient', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Absolute Coefficient', y='Feature', data=feature_importance.head(20), palette='viridis')
        plt.title('Top 20 Feature Importances (Linear Model)', fontsize=16, pad=12)
        plt.xlabel('Absolute Coefficient Value', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def _plot_tree_model_feature_importance(model, feature_names: Dict, save_path: str):
        """Helper to plot feature importance for tree-based models."""
        # Extract feature names
        feature_names_num = feature_names["num_cols"]
        feature_names_cat = feature_names["cat_cols"]
        all_feature_names = np.concatenate([feature_names_num, feature_names_cat])

        # Get feature importances from the tree-based model
        feature_importances = model.feature_importances_

        # Create DataFrame for feature importance
        feature_importance = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': feature_importances
        })

        # Sort by importance
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20), palette='viridis')
        plt.title('Top 20 Feature Importances (Tree Model)', fontsize=16, pad=12)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

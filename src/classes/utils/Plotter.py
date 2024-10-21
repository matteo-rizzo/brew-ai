import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Plotter:
    """
    Handles the creation and saving of various plots such as actual vs predicted values, residuals, and feature importance.
    """

    @staticmethod
    def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray, save_path: str):
        """Generate scatter plot for actual vs predicted values and save to the given path."""
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, color='blue', s=60, edgecolor='black', alpha=0.7, label='Predicted')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
        plt.title('Actual vs Predicted Values', fontsize=18, pad=15)
        plt.xlabel('Actual Values (Tempo di riduzione diacetile)', fontsize=14)
        plt.ylabel('Predicted Values (Tempo di riduzione diacetile)', fontsize=14)
        plt.legend(loc='upper left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_residuals(y_test: np.ndarray, y_pred: np.ndarray, save_path: str):
        """Generate residuals plot and save to the given path."""
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 8))
        sns.histplot(residuals, kde=True, color='darkgreen', bins=30, stat='density', line_kws={'lw': 2})
        plt.title('Residuals Distribution', fontsize=18, pad=15)
        plt.xlabel('Residuals', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_feature_importance(model, feature_names: Dict, save_path: str):
        """Generate feature importance plot and save to the given path."""
        if hasattr(model, 'coef_'):
            Plotter._plot_linear_model_feature_importance(model, feature_names, save_path)
        elif hasattr(model, 'feature_importances_'):
            Plotter._plot_tree_model_feature_importance(model, feature_names, save_path)
        else:
            print(f"Model {type(model).__name__} does not support feature importance plotting.")

    @staticmethod
    def _plot_linear_model_feature_importance(model, feature_names: Dict, save_path: str):
        """Helper to plot feature importance for linear models."""
        feature_names_num = feature_names["num_cols"]
        feature_names_cat = feature_names["cat_cols"]
        all_feature_names = np.concatenate([feature_names_num, feature_names_cat])

        coefficients = model.coef_

        feature_importance = pd.DataFrame({
            'Feature': all_feature_names,
            'Coefficient': coefficients,
            'Absolute Coefficient': np.abs(coefficients)
        }).sort_values(by='Absolute Coefficient', ascending=False)

        plt.figure(figsize=(12, 10))
        sns.barplot(x='Absolute Coefficient', y='Feature', data=feature_importance.head(20), palette='coolwarm')
        plt.title('Top 20 Feature Importances (Linear Model)', fontsize=18, pad=15)
        plt.xlabel('Absolute Coefficient Value', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def _plot_tree_model_feature_importance(model, feature_names: Dict, save_path: str):
        """Helper to plot feature importance for tree-based models."""
        feature_names_num = feature_names["num_cols"]
        feature_names_cat = feature_names["cat_cols"]
        all_feature_names = np.concatenate([feature_names_num, feature_names_cat])

        feature_importances = model.feature_importances_

        feature_importance = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20), palette='coolwarm')
        plt.title('Top 20 Feature Importances (Tree Model)', fontsize=18, pad=15)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_results(results_df: pd.DataFrame, save_path: str):
        """
        Plot comparison of RMSE and R² for all models using bar plots and save them to log_dir.

        :param results_df: DataFrame containing the evaluation metrics for all models
        :param save_path: Directory to save the plots
        """
        # Plot and save RMSE comparison
        if 'RMSE' in results_df.columns:
            plt.figure(figsize=(12, 8))
            sns.barplot(x=results_df.index, y=results_df['RMSE'], palette='Blues_d')
            plt.title('RMSE Comparison of Models', fontsize=18, pad=15)
            plt.ylabel('RMSE', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            rmse_plot_path = os.path.join(save_path, "RMSE_Comparison.png")
            plt.savefig(rmse_plot_path)
            plt.close()

        # Plot and save R² comparison
        if 'R^2' in results_df.columns:
            plt.figure(figsize=(12, 8))
            sns.barplot(x=results_df.index, y=results_df['R^2'], palette='Greens_d')
            plt.title('R^2 Comparison of Models', fontsize=18, pad=15)
            plt.ylabel('R^2 Score', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            r2_plot_path = os.path.join(save_path, "R2_Comparison.png")
            plt.savefig(r2_plot_path)
            plt.close()

    @staticmethod
    def plot_and_save_losses(train_losses: list, val_losses: list, fold: int, save_path: str):
        """
        Plot and save training and validation losses to the specified log directory.

        :param train_losses: List of training losses over epochs.
        :param val_losses: List of validation losses over epochs.
        :param fold: The current fold number for cross-validation.
        :param save_path: Log path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Train and Validation Losses for Fold {fold + 1}')
        plt.legend()

        # Save the plot to the log directory
        plot_file_path = f"{save_path}/loss_plot_fold_{fold + 1}.png"
        plt.savefig(plot_file_path)
        plt.close()

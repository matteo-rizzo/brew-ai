import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.classes.Logger import Logger

# Initialize custom logger
logger = Logger()


class ResultDisplay:
    """
    Handles displaying of results from cross-validation and final evaluation, including plots.
    """

    @staticmethod
    def display_results(cv_results: dict, final_results: dict):
        """
        Display cross-validation and final evaluation results.

        :param cv_results: Dictionary containing cross-validation results
        :param final_results: Dictionary containing final evaluation results
        """
        cv_results_df = pd.DataFrame(cv_results).T
        logger.info("Cross-Validation Results (5-Fold):")
        logger.info(f"{cv_results_df.to_string()}")

        final_results_df = pd.DataFrame(final_results).T
        logger.info("Final Model Evaluation on Hold-out Test Set:")
        logger.info(f"{final_results_df.to_string()}")

        ResultDisplay._plot_results(final_results_df)

    @staticmethod
    def _plot_results(results_df: pd.DataFrame):
        """
        Plot comparison of RMSE and R² for all models using bar plots.

        :param results_df: DataFrame containing the final evaluation metrics for all models
        """
        logger.info("Generating RMSE and R^2 comparison plots...")

        plt.figure(figsize=(12, 6))
        sns.barplot(x=results_df.index, y=results_df['RMSE'])
        plt.title('RMSE Comparison of Models', fontsize=16)
        plt.ylabel('RMSE', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=results_df.index, y=results_df['R^2'])
        plt.title('R^2 Comparison of Models', fontsize=16)
        plt.ylabel('R^2 Score', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        logger.info("RMSE and R^2 comparison plots generated successfully.")

import json
import os
from typing import Dict

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.classes.Logger import Logger

# Initialize custom logger
logger = Logger()


class ResultsSummarizer:
    """
    Handles displaying of results from cross-validation and final evaluation, including plots.
    """

    @staticmethod
    def summarize(log_dir: str):
        """
        Display cross-validation, final evaluation results, and metrics from the JSON files in the log directory.

        :param log_dir: Directory where JSON files with results are saved
        """
        try:
            # Fetch cross-validation results from JSON files
            cv_results = ResultsSummarizer._fetch_results_from_json(log_dir, "_cv_results.json")
            metrics_results = ResultsSummarizer._fetch_results_from_json(log_dir, "_metrics.json")

            # Convert cross-validation results to DataFrame
            if cv_results:
                cv_results_df = pd.DataFrame(cv_results).T
                logger.info("Cross-Validation Results (5-Fold):")
                logger.info(f"{cv_results_df.to_string()}")

            # Convert metrics results to DataFrame
            if metrics_results:
                metrics_results_df = pd.DataFrame(metrics_results).T
                logger.info("Model Evaluation Metrics:")
                logger.info(f"{metrics_results_df.to_string()}")

                # Plot the metrics results and save to log_dir
                ResultsSummarizer._plot_results(metrics_results_df, log_dir)

            # Fetch and display final evaluation results if they exist
            final_results_path = os.path.join(log_dir, "final_evaluation_results.json")
            if os.path.exists(final_results_path):
                final_results = ResultsSummarizer._fetch_json_file(final_results_path)
                final_results_df = pd.DataFrame(final_results).T
                logger.info("Final Model Evaluation on Hold-out Test Set:")
                logger.info(f"{final_results_df.to_string()}")

                # Plot results and save to log_dir
                ResultsSummarizer._plot_results(final_results_df, log_dir)

        except Exception as e:
            logger.error(f"Error displaying results: {e}")

    @staticmethod
    def _fetch_results_from_json(log_dir: str, file_suffix: str) -> Dict:
        """
        Fetch results from all JSON files in the log directory with the specified suffix.

        :param log_dir: Directory containing the JSON files
        :param file_suffix: Suffix of the JSON files to fetch (e.g., "_cv_results.json", "_metrics.json")
        :return: Dictionary with results from all models
        """
        results = {}

        # Iterate over all JSON files in the log directory
        for file_name in os.listdir(log_dir):
            if file_name.endswith(file_suffix):
                model_name = file_name.replace(file_suffix, "")
                file_path = os.path.join(log_dir, file_name)

                # Load the JSON file
                results[model_name] = ResultsSummarizer._fetch_json_file(file_path)

        return results

    @staticmethod
    def _fetch_json_file(file_path: str) -> Dict:
        """
        Load a JSON file and return its content as a dictionary.

        :param file_path: Path to the JSON file
        :return: Dictionary with the file's content
        """
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            return {}

    @staticmethod
    def _plot_results(results_df: pd.DataFrame, log_dir: str):
        """
        Plot comparison of RMSE and R² for all models using bar plots and save them to log_dir.

        :param results_df: DataFrame containing the evaluation metrics for all models
        :param log_dir: Directory to save the plots
        """
        # Plot and save RMSE comparison
        if 'RMSE' in results_df.columns:
            plt.figure(figsize=(12, 6))
            sns.barplot(x=results_df.index, y=results_df['RMSE'])
            plt.title('RMSE Comparison of Models', fontsize=16)
            plt.ylabel('RMSE', fontsize=14)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            # Save plot to log_dir
            rmse_plot_path = os.path.join(log_dir, "RMSE_Comparison.png")
            plt.savefig(rmse_plot_path)
            logger.info(f"RMSE comparison plot saved to {rmse_plot_path}")
            plt.close()

        # Plot and save R² comparison
        if 'R^2' in results_df.columns:
            plt.figure(figsize=(12, 6))
            sns.barplot(x=results_df.index, y=results_df['R^2'])
            plt.title('R^2 Comparison of Models', fontsize=16)
            plt.ylabel('R^2 Score', fontsize=14)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            # Save plot to log_dir
            r2_plot_path = os.path.join(log_dir, "R2_Comparison.png")
            plt.savefig(r2_plot_path)
            logger.info(f"R² comparison plot saved to {r2_plot_path}")
            plt.close()

        logger.info("RMSE and R^2 comparison plots generated and saved successfully.")

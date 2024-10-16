import json
import os
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.classes.utils.Logger import Logger

# Initialize custom logger
logger = Logger()


class CrossValidator:
    """
    Handles the cross-validation process for model evaluation and logs the results.
    """

    @staticmethod
    def perform_cross_validation(grid_search: GridSearchCV, model_name: str, x: pd.DataFrame, y: np.ndarray,
                                 log_dir: str, scoring_metrics: Optional[List[str]] = None) -> GridSearchCV:
        """
        Perform cross-validation using the provided GridSearchCV object, store the results,
        and log the performance.

        :param grid_search: Initialized GridSearchCV object
        :param model_name: Name of the model
        :param x: Features for modeling
        :param y: Target variable
        :param log_dir: Log directory to store results
        :param scoring_metrics: Optional list of scoring metrics to evaluate (default: ['r2'])
        :return: The fitted GridSearchCV object
        """
        if scoring_metrics is None:
            scoring_metrics = ['r2']

        logger.info(f"Starting cross-validation for model: {model_name} with metrics: {scoring_metrics}...")

        try:
            # Fit the GridSearchCV model
            grid_search.fit(x, y)
            logger.info(f"Cross-validation completed for {model_name}.")

            # Process and store the cross-validation results
            cv_results = CrossValidator._extract_cv_results(grid_search, scoring_metrics)
            CrossValidator._save_cv_results_to_json(model_name, cv_results, log_dir)

            return grid_search

        except Exception as e:
            logger.error(f"Error during cross-validation for {model_name}: {str(e)}")
            raise

    @staticmethod
    def _extract_cv_results(grid_search: GridSearchCV, scoring_metrics: List[str]) -> Dict[str, Dict]:
        """
        Extracts cross-validation results from the fitted GridSearchCV object.

        :param grid_search: Fitted GridSearchCV object
        :param scoring_metrics: List of scoring metrics
        :return: Dictionary containing the processed CV results
        """
        logger.info("Extracting cross-validation results...")
        cv_results = grid_search.cv_results_

        model_cv_results = {}
        for metric in scoring_metrics:
            try:
                mean_test_scores = cv_results[f'mean_test_score']
                std_test_scores = cv_results[f'std_test_score']
                params = cv_results['params']

                model_cv_results[metric] = {
                    'mean_test_scores': mean_test_scores.tolist(),
                    'std_test_scores': std_test_scores.tolist(),
                    'params': params
                }

                logger.info(
                    f"Metric: {metric} | Mean test score: {np.mean(mean_test_scores):.4f} ± {np.mean(std_test_scores):.4f}")

            except KeyError:
                logger.warning(f"Metric {metric} not found in the cross-validation results.")

        return model_cv_results

    @staticmethod
    def _save_cv_results_to_json(model_name: str, cv_results: Dict[str, Dict], log_dir: str) -> None:
        """
        Save the cross-validation results to a JSON file in the specified log directory.

        :param model_name: Name of the model
        :param cv_results: Dictionary containing the cross-validation results
        :param log_dir: Directory to save the JSON file
        """
        try:
            # Ensure the log directory exists
            os.makedirs(log_dir, exist_ok=True)

            # Define the path for the JSON file
            file_path = os.path.join(log_dir, f"{model_name}_cv_results.json")

            # Save the cross-validation results to the JSON file
            with open(file_path, 'w') as json_file:
                json.dump(cv_results, json_file, indent=4)

            logger.info(f"Cross-validation results for {model_name} saved to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save cross-validation results for {model_name}: {e}")
            raise

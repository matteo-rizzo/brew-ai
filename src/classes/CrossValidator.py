import json
import os
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV

from src.classes.Logger import Logger

# Initialize custom logger
logger = Logger()


class CrossValidator:
    """
    Handles the cross-validation process for model evaluation.
    """

    @staticmethod
    def perform_cross_validation(grid_search: GridSearchCV, model_name: str, x: pd.DataFrame, y: np.ndarray,
                                 log_dir: str, scoring_metrics: Optional[List[str]] = None,
                                 cv_splits: Optional[int] = 5) -> None:
        """
        Perform cross-validation for the given model and store the results.

        Supports multiple scoring metrics and custom cross-validation splitting strategy.

        :param grid_search: Initialized GridSearchCV object
        :param model_name: Name of the model
        :param x: Features for modeling
        :param y: Target variable
        :param log_dir: Log directory
        :param scoring_metrics: Optional list of scoring metrics to evaluate (default: 'r2')
        :param cv_splits: Number of cross-validation splits (default: 5)
        """
        try:
            # Set default scoring to 'r2' if no other metrics are provided
            if scoring_metrics is None:
                scoring_metrics = ['r2']

            logger.info(f"Starting cross-validation for [bold blue]{model_name}[/bold blue] with {cv_splits} folds...")

            # Initialize dictionary to store all cv results
            model_cv_results = {}

            # Perform cross-validation for each metric
            for metric in scoring_metrics:
                logger.info(f"Evaluating {model_name} with scoring metric: {metric}...")
                cv_scores = cross_val_score(grid_search, x, y, cv=cv_splits, scoring=metric, n_jobs=-1)

                # Store the results in the cv_results dictionary
                cv_results = {f'{metric} (CV)': f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"}
                model_cv_results[metric] = {
                    'mean': cv_scores.mean(),
                    'std_dev': cv_scores.std(),
                    'scores': cv_scores.tolist()
                }

                logger.info(f"[bold green]Completed cross-validation for {model_name} (Metric: {metric}) "
                            f"with mean {metric}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}[/bold green]")

            # Save the cross-validation results to a JSON file
            CrossValidator._save_cv_results_to_json(model_name, model_cv_results, log_dir)

        except Exception as e:
            logger.error(f"Error during cross-validation for {model_name}: {str(e)}")

    @staticmethod
    def _save_cv_results_to_json(model_name: str, cv_results: dict, log_dir: str) -> None:
        """
        Save the cross-validation results to a JSON file.

        :param model_name: Name of the model
        :param cv_results: Dictionary containing the cross-validation results
        :param log_dir: Directory to save the JSON file
        """
        try:
            # Create the log directory if it doesn't exist
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Define the path for the JSON file
            file_path = os.path.join(log_dir, f"{model_name}_cv_results.json")

            # Save the cross-validation results to a JSON file
            with open(file_path, 'w') as json_file:
                json.dump(cv_results, json_file, indent=4)

            logger.info(f"Cross-validation results for {model_name} saved to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save cross-validation results for {model_name}: {e}")

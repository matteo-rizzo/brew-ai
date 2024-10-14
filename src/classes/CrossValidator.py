from typing import Dict, Optional, List

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
    def perform_cross_validation(grid_search: GridSearchCV, model_name: str, x: np.ndarray, y: np.ndarray,
                                 cv_results: Dict, scoring_metrics: Optional[List[str]] = None,
                                 cv_splits: Optional[int] = 5) -> None:
        """
        Perform cross-validation for the given model and store the results.

        Supports multiple scoring metrics and custom cross-validation splitting strategy.

        :param grid_search: Initialized GridSearchCV object
        :param model_name: Name of the model
        :param x: Features for modeling
        :param y: Target variable
        :param cv_results: Dictionary to store cross-validation results
        :param scoring_metrics: Optional list of scoring metrics to evaluate (default: 'r2')
        :param cv_splits: Number of cross-validation splits (default: 5)
        """
        try:
            # Set default scoring to 'r2' if no other metrics are provided
            if scoring_metrics is None:
                scoring_metrics = ['r2']

            logger.info(f"Starting cross-validation for [bold blue]{model_name}[/bold blue] with {cv_splits} folds...")

            # Perform cross-validation for each metric
            for metric in scoring_metrics:
                logger.info(f"Evaluating {model_name} with scoring metric: {metric}...")
                cv_scores = cross_val_score(grid_search, x, y, cv=cv_splits, scoring=metric, n_jobs=-1)

                # Store the results in the cv_results dictionary
                cv_results[model_name] = {f'{metric} (CV)': f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"}

                logger.info(f"[bold green]Completed cross-validation for {model_name} (Metric: {metric}) "
                            f"with mean {metric}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}[/bold green]")

        except Exception as e:
            logger.error(f"Error during cross-validation for {model_name}: {str(e)}")

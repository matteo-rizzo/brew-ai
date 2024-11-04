import json
import os
from typing import Dict

import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.classes.utils.Logger import Logger
from src.config import CV_METRICS

# Initialize custom logger
logger = Logger()


class CrossValidator:
    def __init__(self, grid_search: GridSearchCV):
        """
        Handles the cross-validation process for model evaluation and logs the results.

        :param grid_search: Initialized GridSearchCV object
        """
        self.grid_search = grid_search
        self.scoring_metrics = CV_METRICS

    def perform_cross_validation(self, model_name: str, x: pd.DataFrame, y: pd.Series, log_dir: str):
        """
        Perform cross-validation using the provided GridSearchCV object, store the results,
        and log the performance.

        :param model_name: Name of the model
        :param x: Features for modeling
        :param y: Target variable
        :param log_dir: Log directory to store results
        :return: The fitted GridSearchCV object
        """
        logger.info(f"Starting cross-validation for model: [bold]{model_name}[/bold] with metrics: {self.scoring_metrics}...")

        try:
            # Fit the GridSearchCV model
            self.grid_search.fit(x, y)
            logger.info(f"Cross-validation completed for [bold]{model_name}[/bold].")

            # Process and store the cross-validation results
            cv_results = self._extract_cv_results()
            Logger().log_metrics_table(model_name, metrics=cv_results)
            self._save_cv_results_to_json(model_name, cv_results, log_dir)
            self._save_best_parameters(model_name, log_dir)

            return self.grid_search

        except Exception as e:
            logger.error(f"Error during cross-validation for [bold]{model_name}[/bold]: {str(e)}")
            raise

    def _extract_cv_results(self) -> Dict[str, float]:
        """
        Extracts cross-validation results from the fitted GridSearchCV object.

        :return: Dictionary containing the processed CV results
        """
        logger.info("Extracting cross-validation results...")
        cv_results_df = pd.DataFrame(self.grid_search.cv_results_)
        best_params = self.grid_search.best_params_

        if best_params:
            # Build the query string to find the best results based on the best parameters
            query_conditions = [f'param_{k} == {repr(v)}' for k, v in best_params.items()]
            best_results_df = cv_results_df.query(' & '.join(query_conditions))
        else:
            best_results_df = cv_results_df

        model_cv_results = {}
        for metric in self.scoring_metrics:
            try:
                # Extract mean and std test score for each metric
                mean_test_score = best_results_df[f'mean_test_{metric}'].values[0]
                std_test_score = best_results_df[f'std_test_{metric}'].values[0]

                # Handle negative scoring (e.g., neg_mean_absolute_error)
                if metric.startswith('neg_'):
                    metric = metric[4:]
                    mean_test_score = abs(mean_test_score)

                model_cv_results[f'{metric}_mean'] = mean_test_score
                model_cv_results[f'{metric}_std'] = std_test_score

            except KeyError:
                logger.warning(f"Metric {metric} not found in cross-validation results.")

        return model_cv_results

    @staticmethod
    def _save_cv_results_to_json(model_name: str, cv_results: Dict[str, float], log_dir: str) -> None:
        """
        Save the cross-validation results to a JSON file in the specified log directory.

        :param model_name: Name of the model
        :param cv_results: Dictionary containing the cross-validation results
        :param log_dir: Directory to save the JSON file
        """
        try:
            # Define the path for the JSON file
            file_path = os.path.join(log_dir, f"{model_name}_cv_results.json")

            # Save the cross-validation results to the JSON file
            with open(file_path, 'w') as json_file:
                json.dump(cv_results, json_file, indent=4)

            logger.info(f"Cross-validation results for [bold]{model_name}[/bold] saved to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save cross-validation results for [bold]{model_name}[/bold]: {str(e)}")
            raise

    def _save_best_parameters(self, model_name: str, model_log_dir: str) -> None:
        """
        Save the best parameters for the model to a JSON file in the model-specific subdirectory.

        :param model_name: The name of the model whose parameters are being saved.
        :param model_log_dir: Subdirectory for storing the best parameters.
        """
        try:
            best_params = self.grid_search.best_params_
            logger.info(f"Best parameters for [bold]{model_name}[/bold]: {best_params}")

            # Define the file path to save the parameters as JSON
            params_file_path = os.path.join(model_log_dir, f"{model_name}_best_params.json")

            # Save the best parameters to a JSON file
            with open(params_file_path, 'w') as file:
                json.dump(best_params, file, indent=4)

            logger.info(f"Best parameters saved to {params_file_path}")

        except Exception as e:
            logger.error(f"Failed to save best parameters for [bold]{model_name}[/bold]: {e}")

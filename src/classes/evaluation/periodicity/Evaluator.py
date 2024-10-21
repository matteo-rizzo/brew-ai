import time

import torch

from src.classes.utils.Logger import Logger
from src.classes.utils.MetricsCalculator import MetricsCalculator

logger = Logger()


class Evaluator:
    def __init__(self, model, model_name: str, log_dir: str):
        """
        Evaluator class to handle model evaluation and metric calculation.

        :param model: Trained model to evaluate.
        :param model_name: Name of the model for logging.
        :param log_dir: Directory to save logs and metrics.
        """
        self.model = model
        self.model_name = model_name
        self.log_dir = log_dir

    def evaluate(self, test_data: dict, fold: int) -> dict:
        """
        Evaluate the model on the test set and compute metrics.

        :param test_data: Dictionary containing test data (inputs and target values).
        :param fold: The current fold number in the cross-validation process.
        :return: Dictionary of evaluation metrics (e.g., RMSE, MAE, R², etc.).
        """
        logger.info(f"Evaluating model on test data for Fold {fold + 1}.")
        eval_start_time = time.time()

        self.model.network.eval()
        with torch.no_grad():
            try:
                if self.model_name.startswith("tab"):
                    test_outputs = self.model.predict(test_data['x_num_p'], test_data['x_num_np'], test_data['x_cat'])
                else:
                    test_outputs = self.model.predict(test_data['x_num_p'], test_data['x_num_np'])

                y_pred = test_outputs.cpu().numpy()
                y_test = test_data['y'].cpu().numpy()

                # Calculate and log metrics
                metrics = MetricsCalculator.calculate_metrics(self.model_name, y_test, y_pred, self.log_dir, fold)
                eval_duration = time.time() - eval_start_time

                logger.info(f"Evaluation for Fold {fold + 1} completed in {eval_duration:.2f} seconds.")
                return metrics

            except Exception as e:
                logger.error(f"Error during model evaluation: {e}")
                return {}

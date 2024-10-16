import os

import pandas as pd
from sklearn.base import BaseEstimator

from src.classes.explanations.CounterfactualExplainer import CounterfactualExplainer
from src.classes.utils.Logger import Logger
from src.classes.explanations.ShapExplainer import ShapExplainer

# Initialize custom logger
logger = Logger()


class Explainer:
    """
    A unified explainer class that wraps SHAP and Counterfactual explanations for machine learning models.
    """

    def __init__(self, model: BaseEstimator, x_train: pd.DataFrame, y_train: pd.DataFrame, log_dir: str,
                 model_type: str = "regressor"):
        """
        Initialize the Explainer class with model, training data, and log directory.

        :param model: Trained machine learning model (from GridSearchCV or similar)
        :param x_train: Training data (Pandas DataFrame)
        :param y_train: Training groundtruth (Pandas DataFrame)
        :param log_dir: Directory to save explanation results
        :param model_type: Type of the model, either 'regressor' or 'classifier'
        """
        self.model = model.named_steps['model'] if hasattr(model, 'named_steps') else model
        self.preprocessor = model.named_steps['preprocessor'] if hasattr(model, 'named_steps') else None
        self.x_train, self.y_train = x_train, y_train
        self.log_dir = log_dir
        self.model_type = model_type

        # Initialize SHAP and Counterfactual Handlers
        self.shap_handler = ShapExplainer(
            model=self.model,
            x_train=self.x_train,
            preprocessor=self.preprocessor,
            log_dir=self.log_dir
        )
        self.cf_handler = CounterfactualExplainer(
            model=self.model,
            x_train=self.x_train,
            y_train=self.y_train,
            model_type=self.model_type,
            log_dir=self.log_dir
        )

        # Ensure the log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def generate_shap_explanations(self, x_test: pd.DataFrame, instance_id: str = "shap_explanation") -> None:
        """
        Generate SHAP explanations for the test set and save them to the log directory.

        :param x_test: Test data (Pandas DataFrame)
        :param instance_id: Identifier for the test instance (for saving purposes)
        """
        try:
            logger.info("Generating SHAP explanations...")

            # Generate SHAP values
            self.shap_handler.generate_shap_values()

            # Save SHAP visualizations
            self.shap_handler.plot_summary()
            self.shap_handler.plot_force()

            logger.info(f"SHAP explanations saved successfully for {instance_id}.")

        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")

    def generate_counterfactual_explanations(self, query_instance: pd.DataFrame, total_cfs: int = 5,
                                             instance_id: str = "cf_explanation") -> None:
        """
        Generate counterfactual explanations for a given query instance and save them to the log directory.

        :param query_instance: Single test instance (Pandas DataFrame)
        :param total_cfs: Number of counterfactuals to generate (default: 5)
        :param instance_id: Identifier for the test instance (for saving purposes)
        """
        try:
            logger.info("Generating counterfactual explanations...")

            # Generate counterfactuals
            counterfactuals = self.cf_handler.generate_counterfactuals(query_instance=query_instance,
                                                                       total_CFs=total_cfs)

            # Save counterfactuals to file
            self.cf_handler.save_counterfactuals(counterfactuals=counterfactuals, instance_id=instance_id)

            logger.info(f"Counterfactual explanations saved successfully for {instance_id}.")

        except Exception as e:
            logger.error(f"Error generating counterfactual explanations: {e}")

    def explain(self, x_test: pd.DataFrame, query_instance: pd.DataFrame, total_cfs: int = 5) -> None:
        """
        Generate both SHAP and Counterfactual explanations for the test set and a single query instance.

        :param x_test: Test data (Pandas DataFrame) for SHAP explanations
        :param query_instance: Single instance (Pandas DataFrame) for counterfactual explanations
        :param total_cfs: Number of counterfactuals to generate (default: 5)
        """
        # Generate SHAP explanations
        self.generate_shap_explanations(x_test=x_test, instance_id="shap_explanation")

        # Generate Counterfactual explanations
        self.generate_counterfactual_explanations(query_instance=query_instance, total_cfs=total_cfs,
                                                  instance_id="cf_explanation")

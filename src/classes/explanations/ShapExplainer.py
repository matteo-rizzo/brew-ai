import os
from typing import List

import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.classes.utils.Logger import Logger

logger = Logger()


class ShapExplainer:
    """
    A class to handle SHAP explanations for various machine learning models.
    Supports generating SHAP values and saving visualizations.

    Usage:
        shap_handler = ShapExplainer(model, x_train, preprocessor, log_dir)
        shap_handler.generate_shap_values()
        shap_handler.plot_summary()
        shap_handler.plot_dependence('FeatureName')
    """

    def __init__(self, model: BaseEstimator, x_train: pd.DataFrame, preprocessor: ColumnTransformer,
                 log_dir: str) -> None:
        """
        Initialize the ShapExplainer with the model, training data, and directory to save plots.

        :param model: The trained model
        :param x_train: The original training data before transformation
        :param preprocessor: Preprocessing pipeline to apply to the data
        :param log_dir: Directory to save SHAP visualizations
        """
        self.model = model
        self.x_train = x_train
        self.preprocessor = preprocessor
        self.log_dir = log_dir
        self.shap_values = None

        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Run preprocessor on training data and store the transformed data
        self.x_train_transformed = self.preprocessor.transform(self.x_train)

        # Extract feature names after preprocessing
        self.feature_names = self._extract_feature_names(preprocessor)
        logger.info(f"ShapExplainer initialized with transformed feature names: {self.feature_names}")

    def generate_shap_values(self, sampling_method: str = "kmeans", k: int = 50) -> None:
        """
        Generate SHAP values for the model using the training data. Summarize the background data
        using random sampling or K-Means clustering.

        :param sampling_method: Sampling method for background data ('random' or 'kmeans')
        :param k: Number of samples or clusters to use for background summarization
        """
        try:
            logger.info(f"Generating SHAP values using {sampling_method} sampling with K={k}...")

            # Sample or summarize background data
            if sampling_method == "random":
                background_data = shap.sample(self.x_train_transformed, k)
            elif sampling_method == "kmeans":
                background_data = shap.kmeans(self.x_train_transformed, k)
            else:
                raise ValueError("Invalid sampling method. Choose 'random' or 'kmeans'.")

            # Use KernelExplainer for non-tree models
            explainer = shap.KernelExplainer(self.model.predict, background_data)
            self.shap_values = explainer.shap_values(self.x_train_transformed)

            logger.info("SHAP values generated successfully.")

        except Exception as e:
            logger.error(f"Error generating SHAP values: {e}")
            raise

    def plot_summary(self, plot_type: str = "violin") -> None:
        """
        Create and save a SHAP summary plot.

        :param plot_type: Type of summary plot to generate (default: "dot").
                          Other options include 'bar' or 'violin'.
        """
        try:
            logger.info(f"Generating SHAP summary plot of type {plot_type}...")
            shap.summary_plot(
                self.shap_values,
                self.x_train_transformed,
                feature_names=self.feature_names,
                plot_type=plot_type
            )

            # Save plot
            summary_plot_path = os.path.join(self.log_dir, f"shap_summary_{plot_type}.png")
            plt.savefig(summary_plot_path)
            plt.close()

            logger.info(f"SHAP summary plot saved at {summary_plot_path}")

        except Exception as e:
            logger.error(f"Error generating SHAP summary plot: {e}")
            raise

    def plot_force(self, index: int = 0) -> None:
        """
        Create and save a SHAP force plot for a single instance.

        :param index: The index of the instance in the dataset to generate the force plot for (default: 0)
        """
        try:
            logger.info(f"Generating SHAP force plot for instance {index}...")
            force_plot = shap.force_plot(self.shap_values[index], self.x_train_transformed[index, :])

            # Save plot as HTML
            force_plot_path = os.path.join(self.log_dir, f"shap_force_{index}.html")
            shap.save_html(force_plot_path, force_plot)

            logger.info(f"SHAP force plot saved at {force_plot_path}")

        except Exception as e:
            logger.error(f"Error generating SHAP force plot for instance {index}: {e}")
            raise

    @staticmethod
    def _extract_feature_names(transformer: ColumnTransformer) -> List[str]:
        """
        Extract feature names from a ColumnTransformer or other preprocessor pipeline.

        :param transformer: The preprocessor (e.g., ColumnTransformer)
        :return: A list of feature names
        """
        feature_names = []

        for name, trans, cols in transformer.transformers_:
            if hasattr(trans, 'get_feature_names_out'):
                if isinstance(trans, Pipeline):
                    # Handle transformers that are pipelines (e.g., OneHotEncoder inside a pipeline)
                    final_transformer = trans.steps[-1][1]
                    if hasattr(final_transformer, 'get_feature_names_out'):
                        feature_names.extend(final_transformer.get_feature_names_out(cols))
                    else:
                        feature_names.extend(cols)  # Use original column names if no feature names available
                else:
                    feature_names.extend(trans.get_feature_names_out(cols))
            else:
                feature_names.extend(cols)

        return feature_names

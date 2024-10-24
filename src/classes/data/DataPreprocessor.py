import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.classes.utils.Logger import Logger

logger = Logger()


class DataPreprocessor:
    """
    Handles data preprocessing such as scaling, encoding, and optional PCA for dimensionality reduction.
    """

    def __init__(self, x: pd.DataFrame, y: pd.Series, cat_cols: list):
        """
        Initialize the DataPreprocessor class.

        :param x: Input feature DataFrame
        :param y: Target variable Series
        """
        self.x = x
        self.y = y

        self.num_cols = [col for col in self.x.columns if col not in cat_cols]
        self.cat_cols = cat_cols

        logger.info(
            f"DataPreprocessor initialized with {len(self.num_cols)} numerical and {len(self.cat_cols)} categorical columns.")

    def make_preprocessor(self) -> ColumnTransformer:
        """
        Preprocess the data by scaling numerical features and encoding categorical features.
        Optionally applies PCA for dimensionality reduction.

        :return: Preprocessor pipeline with scaled numerical features, optional PCA,
                 and encoded categorical features
        :rtype: sklearn.compose.ColumnTransformer
        """
        logger.info("Starting data preprocessing: Scaling numerical and encoding categorical features...")

        # Numerical feature processing: scaling and optional PCA
        num_transformers = [('scaler', StandardScaler())]

        # Create a pipeline for numerical transformations
        num_pipeline = Pipeline(steps=num_transformers)

        # Define the preprocessor pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, self.num_cols),  # Apply scaling (and PCA) to numerical columns
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.cat_cols)  # One-hot encode categorical columns
            ]
        )

        logger.info("Data preprocessing complete.")
        return preprocessor

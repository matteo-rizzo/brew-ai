from typing import Union, List, Dict

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.classes.Logger import Logger

logger = Logger()


class DataPreprocessor:
    """
    Handles data preprocessing such as scaling, encoding, and optional PCA for dimensionality reduction.
    """

    def __init__(self, x: pd.DataFrame, y: pd.Series, apply_pca: bool = True,
                 n_components: Union[int, float] = 0.95):
        """
        Initialize the DataPreprocessor class.

        :param x: Input feature DataFrame
        :param y: Target variable Series
        :param apply_pca: Whether to apply PCA for dimensionality reduction (default: True)
        :param n_components: Number of principal components or variance ratio to retain in PCA (default: 0.95)
        """
        self.x = x
        self.y = y
        self.num_cols = x.select_dtypes(include=['float64', 'int64']).columns.tolist()
        self.cat_cols = x.select_dtypes(include=['object', 'category']).columns.tolist()
        self.apply_pca = apply_pca
        self.n_components = n_components

        logger.info(f"DataPreprocessor initialized with {len(self.num_cols)} numerical and "
                    f"{len(self.cat_cols)} categorical columns.")
        if self.apply_pca:
            logger.info(f"PCA will be applied with n_components={n_components}.")

    def preprocess(self) -> ColumnTransformer:
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
        if self.apply_pca:
            logger.info(f"Applying PCA with n_components={self.n_components} for dimensionality reduction.")
            num_transformers.append(('pca', PCA(n_components=self.n_components)))

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

    def get_feature_names(self, preprocessor: ColumnTransformer) -> Dict[str, List[str]]:
        """
        Get the feature names after the preprocessing transformations have been applied.

        :param preprocessor: The fitted preprocessor used in the pipeline
        :return: A dictionary with numerical and categorical feature names
        """
        logger.info("Extracting feature names after preprocessing...")

        try:
            # Get numerical feature names
            if self.apply_pca:
                n_components = preprocessor.named_transformers_['num'].named_steps['pca'].n_components_
                num_feature_names = [f"PC{i + 1}" for i in range(n_components)]
            else:
                num_feature_names = preprocessor.named_transformers_['num'].named_steps['scaler'] \
                    .get_feature_names_out(self.num_cols)

            # Get categorical feature names
            cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(self.cat_cols)

            logger.info(f"Numerical features after processing: {num_feature_names}")
            logger.info(f"Categorical features after encoding: {cat_feature_names}")

            return {"num_cols": num_feature_names, "cat_cols": cat_feature_names}

        except Exception as e:
            logger.error(f"Error extracting feature names: {e}")
            raise

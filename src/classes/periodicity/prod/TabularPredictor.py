from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.classes.periodicity.ModelFactory import ModelFactory
from src.classes.periodicity.models.base.BaseModel import BaseModel
from src.classes.periodicity.models.base.BaseTabModel import BaseTabModel


class TabularPredictor:
    def __init__(self, model_name: str, path_to_model: str, data_config: Dict, device: torch.device):
        """
        Class to handle prediction using a tabular data model.

        :param model_name: Name of the model.
        :param path_to_model: File path to the model weights (.pth file).
        :param data_config: Dictionary containing data configuration parameters.
        :param device: Torch device (CPU or GPU).
        """
        self.model_name = model_name
        self.path_to_model = path_to_model
        self.data_config = data_config
        self.device = device

    def load_model(self, num_input_size: int, cat_input_size: int) -> BaseModel | BaseTabModel:
        """
        Load a PyTorch model from a .pth file.

        :param num_input_size: Number of numerical features.
        :param cat_input_size: Number of categorical features.
        :return: Loaded PyTorch model.
        """
        model = ModelFactory(num_input_size=num_input_size, cat_input_size=cat_input_size).get_model(self.model_name)
        model.network.load_state_dict(torch.load(self.path_to_model, map_location=self.device))
        model.network.eval()
        return model

    def prepare_data(self, x: pd.DataFrame):
        """
        Preprocess the input DataFrame by:
        - Reordering columns
        - Scaling numerical features
        - One-hot encoding categorical features

        :param x: Input DataFrame for prediction.
        :return: Tuple of (x_num_tsr, x_cat_tsr) as Tensors.
        """
        cat_cols = self.data_config.get("cat_cols", [])
        numerical_columns = [col for col in x.columns if col not in cat_cols]
        categorical_columns = [col for col in x.columns if col in cat_cols]

        # Reorder columns: numerical first, then categorical
        x = x[numerical_columns + categorical_columns]

        # Define preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[('scaler', StandardScaler())]), numerical_columns),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
            ]
        )

        # Fit and transform data
        x_transformed = preprocessor.fit_transform(x)

        # Identify column indices for numerical and categorical features post-transformation
        # Note: After transformation, we must rely on the known order.
        # OneHotEncoder expands columns, so we track them by their known order and sizes.
        # We'll handle indices by counting columns after transformation.
        num_cols_count = len(numerical_columns)
        cat_feature_count = x_transformed.shape[1] - num_cols_count

        x_num = x_transformed[:, :num_cols_count]
        x_cat = x_transformed[:, num_cols_count:num_cols_count + cat_feature_count]

        # Convert to Torch tensors
        x_num_tsr = torch.tensor(x_num, dtype=torch.float32).to(self.device)

        # For categorical features after one-hot encoding, they are now binary indicators.
        x_cat_tsr = torch.tensor(x_cat, dtype=torch.float32).to(self.device)

        return x_num_tsr, x_cat_tsr

    def __call__(self, x: pd.DataFrame) -> np.ndarray:
        """
        Perform prediction on the given DataFrame using the loaded model.

        :param x: Input DataFrame.
        :return: Numpy array of model predictions.
        """
        try:
            x_num, x_cat = self.prepare_data(x)

            num_input_size = x_num.shape[1]
            cat_input_size = x_cat.shape[1]

            print(num_input_size, cat_input_size)

            model = self.load_model(num_input_size, cat_input_size)

            with torch.no_grad():
                outputs = model.predict(x_num=x_num, x_cat=x_cat)

            return outputs.cpu().numpy()
        except Exception as e:
            raise ValueError(f"An error occurred during prediction: {str(e)}")

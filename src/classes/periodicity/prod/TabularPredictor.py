import logging
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.classes.periodicity.ModelFactory import ModelFactory
from src.classes.periodicity.models.base.BaseModel import BaseModel
from src.classes.periodicity.models.base.BaseTabModel import BaseTabModel


class TabularPredictor:
    def __init__(self, model_name: str, path_to_model: str, path_to_preprocessor: str, data_config: Dict,
                 device: torch.device):
        """
        Class to handle prediction using a tabular data model.

        :param model_name: Name of the model.
        :param path_to_model: File path to the model weights (.pth file).
        :param path_to_preprocessor: File path to the pre-fitted scikit-learn preprocessor (.joblib file).
        :param data_config: Dictionary containing data configuration parameters.
        :param device: Torch device (CPU or GPU).
        """
        self.model_name = model_name
        self.path_to_model = path_to_model
        self.data_config = data_config
        self.device = device
        # Load the pre-fitted preprocessor during initialization.
        self.preprocessor = self._load_preprocessor(path_to_preprocessor)
        # Load the model structure (weights will be loaded later)
        self.model = self._initialize_model_structure()

    def _load_preprocessor(self, path: str) -> ColumnTransformer:
        """
        Loads the pre-fitted scikit-learn preprocessor from a file.
        """
        logging.info(f"Loading pre-fitted preprocessor from {path}")
        try:
            return joblib.load(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Preprocessor file not found at {path}. Please ensure it was saved during training.")
        except Exception as e:
            raise RuntimeError(f"Failed to load preprocessor from {path}: {e}")

    def _initialize_model_structure(self) -> BaseModel | BaseTabModel:
        """
        Initializes the model structure based on the preprocessor's output features.
        """
        try:
            # Get the number of numerical features from the 'num' part of the pipeline
            num_features = len(self.preprocessor.named_transformers_['num']['scaler'].get_feature_names_out())

            # FINAL FIX: Safely determine the number of categorical features by inspecting
            # the 'transformers_' attribute, which only contains transformers that were
            # actually fitted and applied to columns during the training phase.
            cat_features = 0
            for name, transformer, columns in self.preprocessor.transformers_:
                if name == 'cat' and len(columns) > 0:
                    # If the 'cat' transformer was used, it's now safe to get its feature count.
                    cat_features = len(transformer.get_feature_names_out())
                    break  # Found it, no need to continue loop

        except Exception as e:
            raise RuntimeError(
                f"Could not determine feature counts from the preprocessor. Is it fitted correctly? Error: {e}")

        logging.info(
            f"Initializing model structure with {num_features} numerical and {cat_features} categorical features.")

        # This part of your code seems to expect num_input_size and cat_input_size separately.
        # We will pass the counts we just calculated.
        model = ModelFactory(num_input_size=num_features, cat_input_size=cat_features).get_model(self.model_name)
        model.network.load_state_dict(torch.load(self.path_to_model, map_location=self.device))
        model.network.eval()
        return model

    def prepare_data(self, x: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess the input DataFrame using the pre-loaded preprocessor.

        :param x: Input DataFrame for prediction.
        :return: Tuple of (x_num_tsr, x_cat_tsr) as Tensors.
        """
        logging.info("Transforming input data using the pre-fitted preprocessor.")
        x_transformed = self.preprocessor.transform(x)

        # The number of numerical columns is known from the preprocessor itself.
        num_cols_count = len(self.preprocessor.named_transformers_['num']['scaler'].get_feature_names_out())

        x_num = x_transformed[:, :num_cols_count]
        x_cat = x_transformed[:, num_cols_count:]

        # Convert to Torch tensors
        x_num_tsr = torch.tensor(x_num, dtype=torch.float32).to(self.device)

        # Handle the case where there are no categorical features
        if x_cat.shape[1] == 0:
            x_cat_tsr = torch.empty((x_cat.shape[0], 0), dtype=torch.float32).to(self.device)
        else:
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

            with torch.no_grad():
                # The model's predict method needs to handle these inputs
                outputs = self.model.predict(x_num=x_num, x_cat=x_cat)

            return outputs.cpu().numpy()
        except Exception as e:
            logging.error(f"An error occurred during prediction: {str(e)}", exc_info=True)
            raise ValueError(f"An error occurred during prediction: {str(e)}")

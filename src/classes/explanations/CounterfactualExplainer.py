import json
import os
from typing import Any

import pandas as pd
from dice_ml import Data, Model
from dice_ml import Dice
from sklearn.base import BaseEstimator

from src.classes.utils.Logger import Logger
from src.config import TARGET

# Initialize custom logger
logger = Logger()


class CounterfactualExplainer:
    """
    Handles the generation of counterfactual explanations for machine learning models.
    """

    def __init__(self, model: BaseEstimator, x_train: pd.DataFrame, y_train: pd.DataFrame, model_type: str,
                 log_dir: str, desired_range: tuple = None):
        """
        Initialize the CounterfactualExplainer with a model and training data.

        :param model: Trained machine learning model
        :param x_train: Training dataset for the model
        :param y_train: Training labels for the model
        :param model_type: Type of the model ('regressor' or 'classifier')
        :param log_dir: Directory to save the counterfactual results
        """
        self.model = model
        self.x_train = x_train
        self.model_type = model_type
        self.log_dir = log_dir
        self.desired_range = desired_range

        # Initialize DiCE data interface and model
        self.dice_data = self._init_dice_data_interface(x_train, y_train)
        self.dice_model = self._init_dice_model_interface()

    @staticmethod
    def _init_dice_data_interface(x_train: pd.DataFrame, y_train: pd.DataFrame) -> Data:
        """
        Initialize the DiCE data interface for handling the training data.

        :param x_train: Training dataset for the model
        :param y_train: Training labels for the model
        :return: Initialized DiCE data interface
        """
        logger.info("Initializing DiCE data interface...")

        # DiCE requires specification of continuous features
        continuous_features = x_train.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Create a DiCE Data object
        dice_data = Data(
            dataframe=pd.concat([x_train, y_train], axis=1),
            continuous_features=continuous_features,
            outcome_name=TARGET
        )

        logger.info("DiCE data interface initialized.")
        return dice_data

    def _init_dice_model_interface(self) -> Model:
        """
        Initialize the DiCE model interface for the trained model.

        :return: Initialized DiCE model interface
        """
        logger.info("Initializing DiCE model interface...")

        # Create a DiCE Model object
        backend = 'sklearn'
        dice_model = Model(model=self.model, backend=backend, model_type=self.model_type)

        logger.info("DiCE model interface initialized.")
        return dice_model

    def generate_counterfactuals(self, query_instance: pd.DataFrame, total_CFs: int = 5) -> Any:
        """
        Generate counterfactual explanations for a given query instance.

        :param query_instance: The input instance to generate counterfactuals for
        :param total_CFs: Number of counterfactual explanations to generate (default: 5)
        :return: Counterfactual explanations
        """
        logger.info("Generating counterfactual explanations...")

        # Initialize DiCE for generating counterfactuals
        dice = Dice(self.dice_data, self.dice_model, method="random")

        # Set desired_range for regression tasks if applicable
        if self.model_type == 'regressor' and self.desired_range:
            logger.info(f"Using desired range {self.desired_range} for regression task.")
            counterfactuals = dice.generate_counterfactuals(
                query_instance, total_CFs=total_CFs, desired_range=self.desired_range, verbose=True
            )
        else:
            counterfactuals = dice.generate_counterfactuals(query_instance, total_CFs=total_CFs, verbose=True)

        logger.info("Counterfactual explanations generated.")
        return counterfactuals

    def save_counterfactuals(self, counterfactuals: Any, instance_id: str) -> None:
        """
        Save the generated counterfactuals to the log directory as a JSON file.

        :param counterfactuals: Generated counterfactual explanations
        :param instance_id: Identifier for the query instance
        """
        logger.info(f"Saving counterfactuals for instance {instance_id}...")

        # Define the file path for saving counterfactual explanations
        counterfactuals_file_path = os.path.join(self.log_dir, f"{instance_id}_counterfactuals.json")

        # Convert the counterfactuals to a dictionary
        counterfactuals_dict = counterfactuals.cf_examples_list[0].final_cfs_df.to_dict()

        # Save the counterfactuals to a JSON file
        with open(counterfactuals_file_path, 'w') as json_file:
            json.dump(counterfactuals_dict, json_file, indent=4)

        logger.info(f"Counterfactuals saved to {counterfactuals_file_path}")

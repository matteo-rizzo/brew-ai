from typing import Dict, Tuple

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from src.classes.utils.Logger import Logger
from src.config import RANDOM_SEED, MAX_ITERATIONS


class ModelConfigFactory:
    """
    Handles the creation of model configurations and their corresponding hyperparameter grids.
    """

    def __init__(self, logger: Logger = None):
        """
        Initialize the ModelConfigFactory with models and hyperparameter grids.

        :param logger: Optional custom logger instance.
        """
        self.logger = logger or Logger()
        self.logger.info("Setting up models and hyperparameter grids...")

        # Define a common set of learning rates for consistency
        learning_rates = [0.001, 0.01, 0.05, 0.1]

        self.models_config: Dict[str, Tuple] = {
            'mlpregressor': (
                MLPRegressor(
                    max_iter=MAX_ITERATIONS,
                    learning_rate="adaptive",
                    hidden_layer_sizes=(128, 64),
                    early_stopping=True,
                    random_state=RANDOM_SEED,
                ),
                {
                    'model__activation': ['relu', 'tanh', 'logistic'],
                    'model__solver': ['adam', 'sgd']
                },
            ),
            'elasticnet': (
                ElasticNet(max_iter=MAX_ITERATIONS),
                {
                    'model__alpha': [1e-5, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0],
                    'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                },
            ),
            'random_forest': (
                RandomForestRegressor(n_estimators=MAX_ITERATIONS, random_state=RANDOM_SEED),
                {
                    'model__max_depth': [None, 5, 10, 20, 30],
                    'model__max_features': ['auto', 'sqrt', 'log2'],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4],
                    'model__bootstrap': [True, False],
                },
            ),
            'xgboost': (
                XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1),
                {
                    'model__n_estimators': [100, 200, 300],
                    'model__learning_rate': learning_rates,
                    'model__max_depth': [3, 5, 7, 9],
                    'model__subsample': [0.6, 0.8, 1.0],
                    'model__colsample_bytree': [0.6, 0.8, 1.0],
                    'model__gamma': [0, 0.1, 0.2],
                    'model__reg_alpha': [0, 0.01, 0.1],
                    'model__reg_lambda': [1, 1.5, 2],
                },
            ),
            'lightgbm': (
                LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1),
                {
                    'model__n_estimators': [100, 200, 300],
                    'model__learning_rate': learning_rates,
                    'model__max_depth': [-1, 5, 10, 20],
                    'model__num_leaves': [31, 50, 70],
                    'model__subsample': [0.6, 0.8, 1.0],
                    'model__colsample_bytree': [0.6, 0.8, 1.0],
                    'model__reg_alpha': [0, 0.01, 0.1],
                    'model__reg_lambda': [0, 0.01, 0.1],
                },
            ),
            'catboost': (
                CatBoostRegressor(iterations=MAX_ITERATIONS, random_state=RANDOM_SEED, verbose=0),
                {
                    'model__learning_rate': learning_rates,
                    'model__depth': [4, 6, 8, 10],
                    'model__l2_leaf_reg': [1, 3, 5, 7],
                    'model__bagging_temperature': [0, 1, 3],
                },
            )
        }

    def get_model_configuration(self, model_name: str) -> Tuple:
        """
        Retrieve the model and its hyperparameters based on the model name.

        :param model_name: The name of the model to retrieve.
        :return: A tuple containing the model instance and hyperparameter grid.
        :raises ValueError: If the model name is not available.
        """
        model_name = model_name.lower()
        try:
            return self.models_config[model_name]
        except KeyError:
            available_models = ', '.join(self.models_config.keys())
            self.logger.error(
                f"Model '{model_name}' is not available. Available models are: {available_models}"
            )
            raise ValueError(
                f"Model '{model_name}' is not available. Available models are: {available_models}"
            )

    def get_model_configurations(self) -> Dict[str, Tuple]:
        """
        Retrieve all models and their associated hyperparameter grids.

        :return: Dictionary with models and their hyperparameter grids.
        """
        return self.models_config

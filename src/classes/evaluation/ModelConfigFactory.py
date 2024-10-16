from typing import Dict, Tuple

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from src.classes.utils.Logger import Logger
from src.settings import RANDOM_SEED

# Initialize custom logger
logger = Logger()


class ModelConfigFactory:
    """
    Handles the creation of model configurations and their corresponding hyperparameter grids.
    """

    def __init__(self):
        logger.info("Setting up models and hyperparameter grids...")
        self.models_config: Dict[str, Tuple] = {
            # Linear Regression: No hyperparameters to tune
            'linear_regression': (LinearRegression(), {}),

            # Ridge Regression: Alpha values for regularization tuning
            'ridge': (
                Ridge(),
                {
                    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    'model__solver': ['lsqr', 'sparse_cg', 'sag']
                }
            ),

            # Lasso Regression: Increased max_iter and tol to improve convergence
            'lasso': (
                Lasso(max_iter=5000),
                {
                    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'model__tol': [0.001, 0.01]
                }
            ),

            # ElasticNet: Combined Lasso and Ridge with l1_ratio tuning
            'elasticnet': (
                ElasticNet(max_iter=5000),
                {
                    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
            ),

            # Random Forest: n_estimators, max_depth, and min_samples_split for tuning
            'random_forest': (
                RandomForestRegressor(random_state=RANDOM_SEED),
                {
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [10, 20, 30, None],
                    'model__max_features': ['log2', 'sqrt'],
                    'model__min_samples_split': [2, 5, 10]
                }
            ),

            # Gradient Boosting: Added subsample for regularization
            'gradient_boosting': (
                GradientBoostingRegressor(random_state=RANDOM_SEED),
                {
                    'model__n_estimators': [100, 200, 300],
                    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7],
                    'model__subsample': [0.7, 0.8, 1.0]
                }
            ),

            # XGBoost: Fine-tuned learning rate, n_estimators, and max_depth
            'xgboost': (
                XGBRegressor(random_state=RANDOM_SEED, use_label_encoder=False),
                {
                    'model__n_estimators': [100, 200, 300],
                    'model__learning_rate': [0.01, 0.05, 0.1],
                    'model__max_depth': [3, 5, 7],
                    'model__subsample': [0.7, 0.8, 1.0],
                    'model__colsample_bytree': [0.7, 0.8, 1.0]
                }
            ),

            # LightGBM: Tuning number of leaves, depth, and learning rate
            'lightgbm': (
                LGBMRegressor(random_state=RANDOM_SEED),
                {
                    'model__n_estimators': [100, 200, 300],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__max_depth': [-1, 5, 10],
                    'model__num_leaves': [31, 50, 100]
                }
            ),

            # CatBoost: Iterations and depth tuning
            'catboost': (
                CatBoostRegressor(random_state=RANDOM_SEED, silent=True),
                {
                    'model__iterations': [100, 200, 300],
                    'model__learning_rate': [0.01, 0.05, 0.1],
                    'model__depth': [4, 6, 10]
                }
            ),

            # MLPRegressor: Tuning the hidden layer sizes, activation function, and learning rate
            'mlpregressor': (
                MLPRegressor(random_state=RANDOM_SEED, max_iter=1000),
                {
                    'model__hidden_layer_sizes': [(64,), (128,), (128, 64), (128, 64, 32)],  # Different architectures
                    'model__activation': ['relu', 'tanh'],  # Activation functions
                    'model__learning_rate_init': [0.001, 0.01, 0.05],  # Initial learning rates
                    'model__solver': ['adam', 'lbfgs'],  # Solvers for weight optimization
                    'model__alpha': [0.0001, 0.001, 0.01],  # L2 penalty (regularization term)
                }
            )
        }

    def get_model_configuration(self, model_name: str) -> Tuple:
        """
        Retrieve the model and its hyperparameters based on the model name.

        :param model_name: The name of the model to retrieve
        :return: A tuple containing the model instance and hyperparameter grid
        """
        model_name = model_name.lower()  # Ensure lowercase for consistent access
        if model_name in self.models_config:
            return self.models_config[model_name]
        else:
            logger.error(f"Model {model_name} is not available.")
            raise ValueError(f"Model {model_name} is not available.")

    def get_model_configurations(self) -> Dict[str, Tuple]:
        """
        Retrieve all models and their associated hyperparameter grids.

        :return: Dictionary with models and their hyperparameter grids
        """
        return self.models_config

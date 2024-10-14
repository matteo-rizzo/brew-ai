from typing import Dict, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.classes.Logger import Logger

# Initialize custom logger
logger = Logger()


class PipelineBuilder:
    """
    Handles the creation of model pipelines and GridSearchCV instances.
    """

    @staticmethod
    def get_model_parameters() -> Dict[str, Tuple]:
        """
        Define the models and their hyperparameter grids for tuning.

        :return: Dictionary with models and associated hyperparameter grids
        """
        logger.info("Setting up models and hyperparameter grids...")
        return {
            # Linear Regression has no hyperparameters to tune
            'Linear Regression': (LinearRegression(), {}),

            # Ridge Regression: Tuned alpha values for a broad range of regularization
            'Ridge Regression': (
                Ridge(),
                {
                    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    'model__solver': ['lsqr', 'sparse_cg', 'sag', 'lbfgs']
                }
            ),

            # Lasso Regression: Increased max_iter and added tol to aid convergence
            'Lasso Regression': (
                Lasso(max_iter=5000),
                {
                    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'model__tol': [0.001, 0.01]
                }
            ),

            # ElasticNet: Balance between Lasso and Ridge, with a wider range for l1_ratio
            'ElasticNet': (
                ElasticNet(max_iter=5000),
                {
                    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
            ),

            # Random Forest: Expanded n_estimators, added max_depth for tree size control
            'Random Forest': (
                RandomForestRegressor(random_state=42),
                {
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [10, 20, 30, None],
                    'model__max_features': ['auto', 'sqrt'],
                    'model__min_samples_split': [2, 5, 10]
                }
            ),

            # Gradient Boosting: Added subsample for better regularization
            'Gradient Boosting': (
                GradientBoostingRegressor(random_state=42),
                {
                    'model__n_estimators': [100, 200, 300],
                    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7],
                    'model__subsample': [0.7, 0.8, 1.0]
                }
            ),

            # XGBoost: Fine-tuned learning rates, n_estimators, and max_depth
            'XGBoost': (
                XGBRegressor(random_state=42, use_label_encoder=False),
                {
                    'model__n_estimators': [100, 200, 300],
                    'model__learning_rate': [0.01, 0.05, 0.1],
                    'model__max_depth': [3, 5, 7],
                    'model__subsample': [0.7, 0.8, 1.0],
                    'model__colsample_bytree': [0.7, 0.8, 1.0]
                }
            )
        }

    @staticmethod
    def create_pipeline_and_grid_search(preprocessor: ColumnTransformer, model, param_grid: Dict) -> GridSearchCV:
        """
        Create a pipeline with preprocessing and the model, then initialize GridSearchCV.

        :param preprocessor: Preprocessing pipeline
        :param model: Machine learning model
        :param param_grid: Hyperparameter grid for the model
        :return: Initialized GridSearchCV object
        """
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        return GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, return_train_score=True)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.classes.data.DataPreprocessor import DataPreprocessor
from src.classes.explanations.Explainer import Explainer
from src.classes.utils.Logger import Logger
from src.classes.evaluation.grid_search.ModelConfigFactory import ModelConfigFactory
from src.functions.utils import make_model_subdirectory, make_log_dir, load_data, load_best_params
from src.config import RANDOM_SEED, TEST_SIZE, APPLY_PCA

# Initialize the custom logger
logger = Logger()


def main():
    """
    Main function to load data, model, and run the explainer on the model and data.
    """
    try:
        # Configuration
        model_name = 'random_forest'  # Use lowercase for consistent model names

        # Create directories for logging
        log_dir = make_model_subdirectory(model_name, log_dir=make_log_dir(log_type="explanations"))
        logger.info(f"Created log directory for {model_name}: {log_dir}")

        # Load dataset
        logger.info("Loading dataset...")
        x, y = load_data()

        # Data Preprocessing
        logger.info(f"Preprocessing the data with PCA={'enabled' if APPLY_PCA else 'disabled'}...")
        preprocessor = DataPreprocessor(x, y, apply_pca=APPLY_PCA).preprocess()

        # Split data into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

        # Load model from ModelConfigFactory
        logger.info(f"Loading model configuration for {model_name}...")
        model_factory = ModelConfigFactory()
        model, _ = model_factory.get_model_configuration(model_name)

        # Load best model parameters if available
        logger.info(f"Loading best parameters for {model_name} if available...")
        best_params = load_best_params(model_name, log_dir)
        if best_params:
            model.set_params(**best_params)
            logger.info(f"Best parameters applied to {model_name}: {best_params}")
        else:
            logger.info(f"Using default parameters for {model_name}.")

        # Create a pipeline with the preprocessor and the model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        # Train the pipeline
        logger.info(f"Training model {model_name}...")
        pipeline.fit(x_train, y_train)
        logger.info(f"Training completed for {model_name}.")

        # Initialize and run the Explainer
        logger.info(f"Initializing explainer for {model_name}...")
        explainer = Explainer(model=pipeline, x_train=x_train, y_train=y_train, log_dir=log_dir, model_type='regressor')

        # Pick the first instance from the test set as an example for explanations
        query_instance = pd.DataFrame([x_test.iloc[0]])
        logger.info(f"Running explanation process on test instance...")
        explainer.explain(x_test=x_test, query_instance=query_instance)

        logger.info("Explanation process completed successfully.")

    except FileNotFoundError as fnf_error:
        logger.error(f"File not found error: {fnf_error}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    main()

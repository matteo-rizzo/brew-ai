import argparse
import warnings

from sklearn.linear_model._cd_fast import ConvergenceWarning

from src.classes.data.DataPreprocessor import DataPreprocessor
from src.classes.data.DatasetLoader import DatasetLoader
from src.classes.grid_search.ExperimentHandler import ExperimentHandler
from src.classes.utils.Logger import Logger
from src.config import BASE_LOG_DIR, DATASET_ID
from src.functions.utils import make_log_dir

# Suppress convergence warnings and other unnecessary warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Initialize the custom logger globally
logger = Logger()


def main(dataset_id: str):
    """Main function to coordinate the data loading, preprocessing, and model evaluation."""
    try:
        logger.info("Starting the models training and evaluation GridSearchCV pipeline...")

        log_dir = make_log_dir(BASE_LOG_DIR, log_type=f"{dataset_id}__grid_search")
        logger.info(f"Log directory created at: {log_dir}")

        logger.info(f"Loading dataset with ID: {dataset_id}...")
        x, y = DatasetLoader().load_dataset(dataset_id)

        logger.info("Preprocessing the dataset...")
        cat_cols = DatasetLoader().get_dataset_config(dataset_id)["cat_cols"]
        preprocessor = DataPreprocessor(x, y, cat_cols).make_preprocessor()

        logger.info("Initializing model training and evaluation...")
        experiment_handler = ExperimentHandler(x, y, preprocessor, log_dir)
        experiment_handler.run_experiment()
        logger.info("Models training and evaluation GridSearchCV completed successfully.")
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the process. {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation.")
    parser.add_argument('--dataset', type=str, default=DATASET_ID,
                        help="Specify the dataset ID. Defaults to the global DATASET_ID.")
    args = parser.parse_args()
    main(args.dataset)

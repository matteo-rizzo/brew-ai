import warnings

from src.classes.data.DataPreprocessor import DataPreprocessor
from src.classes.evaluation.ExperimentHandler import ExperimentHandler
from src.classes.evaluation.ResultsSummarizer import ResultsSummarizer
from src.classes.utils.Logger import Logger
from src.functions.utils import make_log_dir, load_data
from src.settings import BASE_LOG_DIR, APPLY_PCA

# Ignore warnings to keep the output clean
warnings.filterwarnings('ignore')

# Initialize the custom logger
logger = Logger()


def main():
    try:
        logger.info("Starting the model training and evaluation pipeline...")

        # Set the log directory
        log_dir = make_log_dir(BASE_LOG_DIR)

        # Load the dataset
        logger.info("Loading dataset...")
        x, y = load_data()

        # Step 1: Data Preprocessing
        logger.info(f"Preprocessing the data with PCA={'enabled' if APPLY_PCA else 'disabled'}...")
        data_preprocessor = DataPreprocessor(x, y, apply_pca=APPLY_PCA)
        preprocessor = data_preprocessor.preprocess()

        # Step 2: Model Training and Evaluation
        logger.info("Initializing model training and evaluation...")
        experiment_handler = ExperimentHandler(x, y, preprocessor, log_dir)
        experiment_handler.run_experiment()

        # Step 3: Summarize Results
        logger.info("Summarizing the results...")
        ResultsSummarizer.summarize(log_dir)

        logger.info("Model training and evaluation completed successfully.")

    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
    except Exception as e:
        logger.error(f"An error occurred during the process: {e}")
        raise


if __name__ == '__main__':
    main()

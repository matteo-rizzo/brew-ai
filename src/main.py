import os
import time
import warnings

import pandas as pd

from src.classes.DataPreprocessor import DataPreprocessor
from src.classes.Logger import Logger
from src.classes.ExperimentHandler import ExperimentHandler
from src.classes.ResultsSummarizer import ResultsSummarizer

warnings.filterwarnings('ignore')

# Initialize the custom logger
logger = Logger()


def main():
    """
    Main function to handle the end-to-end model training and evaluation process.
    Loads the dataset, processes it, trains multiple models, and displays results.
    """

    logger.info("Loading dataset...")

    # Set the log directory
    log_dir = os.path.join("logs", f"experiment_{time.time()}")

    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        logger.info(f"Created log directory at {log_dir}")

    # Load the dataset
    df = pd.read_csv('dataset.csv', index_col=False)
    target_column = "Tempo di riduzione diacetile"
    x = df.drop(columns=[target_column, "Unnamed: 0"])
    y = df[target_column]

    logger.info("Starting model training and evaluation...")

    # Step 1: Data Preprocessing
    logger.info("Preprocessing the data...")
    data_preprocessor = DataPreprocessor(x, y, apply_pca=True)
    preprocessor = data_preprocessor.preprocess()

    # Step 2: Model Training and Evaluation
    logger.info("Initializing model training...")
    model_handler = ExperimentHandler(x, y, preprocessor, log_dir)
    model_handler.train_and_evaluate()

    # Step 3: Display Results
    logger.info("Displaying results...")
    ResultsSummarizer.summarize(log_dir)

    logger.info("Model training and evaluation completed.")


if __name__ == '__main__':
    main()

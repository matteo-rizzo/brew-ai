import json
import os
import time

from src.classes.utils.Logger import Logger
from src.config import BASE_LOG_DIR, BENCHMARK, SUITE_ID

# Initialize custom logger
logger = Logger()


def load_best_params(model_name: str, log_dir: str) -> dict:
    """
    Load best parameters from a JSON file if available.

    :param model_name: Name of the model to load parameters for
    :param log_dir: Directory where the JSON file is stored
    :return: Dictionary of the best parameters or an empty dict if not found
    """
    params_file = os.path.join(log_dir, f"{model_name}_best_params.json")

    if not os.path.exists(params_file):
        logger.warning(f"No best parameters found for {model_name}. Using default parameters.")
        return {}

    try:
        with open(params_file, 'r') as file:
            best_params = json.load(file)
        logger.info(f"Successfully loaded best parameters for {model_name}.")
        return best_params
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file {params_file}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading best parameters from {params_file}: {e}")
        raise


def make_log_dir(base_dir: str = BASE_LOG_DIR, log_type: str = "experiment") -> str:
    """
    Create a log directory to store the results of the current run.

    :param base_dir: The base directory for logging (default is from settings)
    :param log_type: The type of log that is used as prefix for the folder
    :return: The path to the newly created log directory
    """
    timestamp = int(time.time())
    log_dir = os.path.join(base_dir, f"{log_type}_{timestamp}")

    try:
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Created log directory: {log_dir}")
        return log_dir
    except OSError as e:
        logger.error(f"Failed to create log directory at {log_dir}. Error: {e}")
        raise


def make_model_subdirectory(model_name: str, log_dir: str) -> str:
    """
    Create a subdirectory within the log directory for a specific model.

    :param model_name: Name of the model (will be used to name the subdirectory)
    :param log_dir: Path to the main log directory
    :return: Path to the created subdirectory for the model
    """
    model_log_dir = os.path.join(log_dir, model_name.replace(' ', '_').lower())
    try:
        os.makedirs(model_log_dir, exist_ok=True)
        logger.info(f"Created subdirectory for {model_name} at {model_log_dir}")
        return model_log_dir
    except OSError as e:
        logger.error(f"Failed to create model subdirectory at {model_log_dir}. Error: {e}")
        raise

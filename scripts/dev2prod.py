import argparse

from src.classes.data.DatasetLoader import DatasetLoader
from src.classes.periodicity.ExperimentHandler import ExperimentHandler
from src.classes.utils.Logger import Logger
from src.config import MODEL, DATASET_ID
from src.functions.utils import make_log_dir

logger = Logger()


def main(model_type: str, dataset_id: str):
    try:
        logger.info(f"Starting the production training process with model: {model_type}")

        # Prepare directories for logging
        log_dir = make_log_dir(log_type=f"{dataset_id}__{model_type}_prod")
        logger.info(f"Logging directory created at {log_dir}")

        # Load data
        logger.info("Loading data...")
        x, y = DatasetLoader().load_dataset(dataset_id)
        logger.info(f"Data loaded successfully. Shape: x={x.shape}, y={y.shape}")

        # Run the experiment
        ExperimentHandler(model_type, dataset_id, x, y, log_dir, mode="prod").run_experiment()
        logger.info("Production training process completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during production training: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model production training.")
    parser.add_argument('--model', type=str, default=MODEL,
                        help="Specify the model. Defaults to the global MODEL_PERIODICITY.")
    parser.add_argument('--dataset', type=str, default=DATASET_ID,
                        help="Specify the dataset ID. Defaults to the global DATASET_ID.")
    args = parser.parse_args()
    main(args.model, args.dataset)

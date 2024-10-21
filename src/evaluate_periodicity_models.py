import argparse

from src.classes.evaluation.periodicity.ExperimentHandler import ExperimentHandler
from src.classes.utils.Logger import Logger
from src.config import MODEL_PERIODICITY
from src.functions.utils import load_data, make_model_subdirectory, make_log_dir

logger = Logger()


def main(model_type: str):
    try:
        logger.info(f"Starting the evaluation process with model: {model_type}")

        # Prepare directories for logging
        log_dir = make_model_subdirectory(
            model_name=model_type,
            log_dir=make_log_dir(log_type="evaluation_periodicity")
        )
        logger.info(f"Logging directory created at {log_dir}")

        # Load data
        logger.info("Loading data...")
        x, y = load_data()
        logger.info(f"Data loaded successfully. Shape: x={x.shape}, y={y.shape}")

        # Run the experiment
        ExperimentHandler(model_name=model_type, x=x, y=y, log_dir=log_dir).run_experiment()
        logger.info("Evaluation process completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run periodicity model evaluation.")
    parser.add_argument('--model_type', type=str, default=MODEL_PERIODICITY,
                        help="Specify the model periodicity. Defaults to the global MODEL_PERIODICITY.")
    args = parser.parse_args()
    main(args.model_type)

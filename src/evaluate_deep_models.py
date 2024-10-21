import argparse
import warnings

from src.classes.data.DataPreprocessor import DataPreprocessor
from src.classes.evaluation.deep_learning.CrossValidator import CrossValidator
from src.classes.evaluation.deep_learning.ModelFactory import ModelFactory
from src.classes.utils.Logger import Logger
from src.functions.utils import load_data, make_log_dir, make_model_subdirectory
from src.config import APPLY_PCA, MODEL_DEEP

# Ignore warnings to keep the output clean
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)

# Logger setup
logger = Logger()


def main(model_type):
    """
    Main function to run the unified framework
    :param model_type: str - Type of model to use (e.g., 'TabNet', 'MLP', etc.)
    """

    log_dir = make_model_subdirectory(model_name=model_type, log_dir=make_log_dir(log_type="evaluation_deep"))

    # Load the dataset
    logger.info("Loading dataset...")
    x, y = load_data()

    # Data Preprocessing
    logger.info(f"Preprocessing the data with PCA={'enabled' if APPLY_PCA else 'disabled'}...")
    data_preprocessor = DataPreprocessor(x, y, apply_pca=APPLY_PCA)
    preprocessor = data_preprocessor.preprocess()

    idx_num = [x.columns.get_loc(col) for col in x.select_dtypes(include=['float64', 'int64']).columns.tolist()]
    idx_cat = [x.columns.get_loc(col) for col in x.select_dtypes(include=['object', 'category']).columns.tolist()]

    x = preprocessor.fit_transform(x)

    # Initialize the model
    model = ModelFactory(model_name=model_type).get_model()

    # Train the model
    trainer = CrossValidator(model, x, idx_num, idx_cat, y.to_numpy(), log_dir)
    trainer.cross_validate()


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run deep_learning model cross-validation.")
    parser.add_argument('--model_type', type=str, default=MODEL_DEEP)

    args = parser.parse_args()

    # Call main with the parsed model type
    main(model_type=args.model_type)

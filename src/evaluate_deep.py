from src.classes.data.DataPreprocessor import DataPreprocessor
from src.classes.evaluation.deep.CrossValidator import CrossValidator
from src.classes.evaluation.deep.ModelFactory import ModelFactory
from src.classes.utils.Logger import Logger
from src.functions.utils import load_data, make_log_dir, make_model_subdirectory
from src.settings import APPLY_PCA, MODEL_DEEP

# Logger setup
logger = Logger()


def main():
    """
    Main function to run the unified framework
    """

    log_dir = make_model_subdirectory(model_name=MODEL_DEEP, log_dir=make_log_dir(log_type="evaluation_deep"))

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
    model = ModelFactory(model_name=MODEL_DEEP).get_model()

    # Train the model
    trainer = CrossValidator(model, x, idx_num, idx_cat, y.to_numpy(), log_dir)
    trainer.cross_validate()


if __name__ == "__main__":
    main()

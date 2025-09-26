# Append required paths
import sys

sys.path.append(".")
sys.path.append("src")

import argparse

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from src.classes.data.DatasetLoader import DatasetLoader
from src.classes.periodicity.ExperimentHandler import ExperimentHandler
from src.classes.utils.Logger import Logger
from src.classes.utils.PeriodicityDetector import PeriodicityDetector
from src.config import MODEL, DATASET_ID, CLASSIFICATION
from src.functions.utils import make_log_dir

logger = Logger()


def main(model_type: str, dataset_id: str):
    try:
        logger.info(f"Starting the evaluation process with model: {model_type}")

        # --- 1. DIRECTORY AND CONFIG SETUP ---
        log_dir = make_log_dir(log_type=f"{dataset_id}__{model_type}_evaluation")
        logger.info(f"Logging to {log_dir}")

        dataset_loader = DatasetLoader()
        x, y = dataset_loader.load_dataset(dataset_id)
        dataset_config = dataset_loader.get_dataset_config(dataset_id)
        logger.info(f"Data loaded successfully. Shape: x={x.shape}, y={y.shape}")

        # --- 2. FEATURE DEFINITION AND REORDERING ---
        cat_cols = dataset_config.get("cat_cols", [])
        feature_selection = dataset_config.get("feature_selection", x.columns)
        numerical_cols = [col for col in x.columns if (col not in cat_cols) and (col in feature_selection)]
        categorical_cols = [col for col in x.columns if (col in cat_cols) and (col in feature_selection)]

        # Reorder columns so numerical are first, then categorical. This is crucial for consistent indexing.
        ordered_columns = numerical_cols + categorical_cols
        x = x[ordered_columns]
        logger.info(f"Columns reordered. {len(numerical_cols)} numerical, {len(categorical_cols)} categorical.")

        # --- 3. PERIODICITY DETECTION (on original, pre-transformed data) ---
        logger.info("Detecting periodic features...")
        original_periodic_indices = []
        for column in numerical_cols:
            series = x[column].values
            if PeriodicityDetector().detect_periodicity_acf(series):
                original_periodic_indices.append(x.columns.get_loc(column))

        logger.info(f"Detected {len(original_periodic_indices)} periodic features.")

        # --- 4. PREPROCESSOR CREATION AND FITTING ---
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ],
            remainder='passthrough'
        )

        if CLASSIFICATION:
            # If target needs encoding, do it here based on the training set
            le = LabelEncoder()
            y = le.fit_transform(y)
            logger.info("Target variable encoded.")

        logger.info("Fitting the preprocessor on the training data...")
        preprocessor.fit(x)
        logger.info("Preprocessor fitted successfully.")

        # --- 5. DATA TRANSFORMATION AND FINAL INDEX CALCULATION ---
        logger.info("Transforming data with the fitted preprocessor...")
        x = preprocessor.transform(x)

        # The preprocessor changes the number of columns. We must calculate the new indices.
        num_features_transformed = len(numerical_cols)

        if categorical_cols:
            cat_features_transformed = \
                preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).shape[0]
        else:
            cat_features_transformed = 0

        idx_num_new = list(range(num_features_transformed))
        idx_cat_new = list(range(num_features_transformed, num_features_transformed + cat_features_transformed))

        # The periodic indices were based on the original dataframe. Since we ordered numerical columns
        # first, their new indices are the same as their old ones within the numerical block.
        idx_periodic_new = [idx for idx in original_periodic_indices if idx < len(numerical_cols)]
        idx_non_periodic_new = [idx for idx in idx_num_new if idx not in idx_periodic_new]

        logger.info(
            f"Transformed data created. New indices: Num={len(idx_num_new)}, Cat={len(idx_cat_new)}, Per={len(idx_periodic_new)}")

        # --- 6. RUN EXPERIMENT ---
        logger.info("Initializing and running the Experiment Handler...")
        experiment_handler = ExperimentHandler(
            model_name=model_type,
            dataset_config=dataset_config,
            log_dir=log_dir,
            mode="cv"
        )
        # Pass the correctly typed pandas objects
        experiment_handler.run_experiment(
            x=x,
            y=y,
            idx_num=idx_num_new,
            idx_cat=idx_cat_new,
            idx_periodic=idx_periodic_new,
            idx_non_periodic=idx_non_periodic_new
        )
        logger.info("Evaluation process completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation.")
    parser.add_argument('--model', type=str, default=MODEL,
                        help="Specify the model. Defaults to the global MODEL_PERIODICITY.")
    parser.add_argument('--dataset', type=str, default=DATASET_ID,
                        help="Specify the dataset ID. Defaults to the global DATASET_ID.")
    args = parser.parse_args()
    main(args.model, args.dataset)

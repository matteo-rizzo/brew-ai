import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.classes.data.DatasetLoader import DatasetLoader
from src.classes.periodicity.ExperimentHandler import ExperimentHandler  # Using the new refactored class
from src.classes.utils.Logger import Logger
from src.classes.utils.PeriodicityDetector import PeriodicityDetector  # Needed for orchestration
from src.config import MODEL, DATASET_ID, CLASSIFICATION
from src.functions.utils import make_log_dir

logger = Logger()


def main(model_type: str, dataset_id: str):
    try:
        logger.info(f"Starting the production training process with model: {model_type}")

        # --- 1. DIRECTORY AND CONFIG SETUP ---
        log_dir = make_log_dir(log_type=f"{dataset_id}__{model_type}_prod")
        prod_dir = os.path.join(os.path.dirname(log_dir), "prod_artifacts")
        os.makedirs(prod_dir, exist_ok=True)
        logger.info(f"Logs: {log_dir} | Production artifacts: {prod_dir}")

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
        x_num_cols_for_periodicity = [col for col in numerical_cols if col != 'month']
        for column in x_num_cols_for_periodicity:
            series = x[column].values
            if PeriodicityDetector().detect_periodicity_acf(series):
                original_periodic_indices.append(x.columns.get_loc(column))

        logger.info(f"Detected {len(original_periodic_indices)} periodic features.")

        # --- 4. PREPROCESSOR CREATION AND FITTING ---
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[('scaler', StandardScaler())]), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ],
            remainder='passthrough'
        )

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        if CLASSIFICATION:
            # If target needs encoding, do it here based on the training set
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            logger.info("Target variable encoded.")

        logger.info("Fitting the preprocessor on the training data...")
        preprocessor.fit(X_train)
        logger.info("Preprocessor fitted successfully.")

        preprocessor_path = os.path.join(prod_dir, "preprocessor.joblib")
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to: {preprocessor_path}")

        # --- 5. DATA TRANSFORMATION AND FINAL INDEX CALCULATION ---
        logger.info("Transforming data with the fitted preprocessor...")
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Combine for final production training using np.concatenate
        X_prod_np = np.concatenate((X_train_transformed, X_test_transformed), axis=0)
        y_prod_np = np.concatenate((
            y_train if isinstance(y_train, np.ndarray) else y_train.values,
            y_test if isinstance(y_test, np.ndarray) else y_test.values
        ), axis=0)

        # Convert the final NumPy arrays back to pandas DataFrame and Series
        # to match the expected input types of ProdHandler and DataSplitter.
        X_prod = pd.DataFrame(X_prod_np)
        y_prod = pd.Series(y_prod_np)

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
            mode="prod"
        )
        # Pass the correctly typed pandas objects
        experiment_handler.run_experiment(
            x=X_prod,
            y=y_prod,
            idx_num=idx_num_new,
            idx_cat=idx_cat_new,
            idx_periodic=idx_periodic_new,
            idx_non_periodic=idx_non_periodic_new
        )
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

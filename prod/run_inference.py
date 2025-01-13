import argparse
import sys

import pandas as pd
import torch

sys.path.append('../')

from src.classes.data.DatabaseManager import DatabaseManager
from src.classes.periodicity.prod.TabularPredictor import TabularPredictor
from src.classes.utils.ConfigManager import ConfigManager


def main():
    # ----------------------
    # 1) Parse Arguments
    # ----------------------
    parser = argparse.ArgumentParser(description="Run tabular predictions with a specified model.")
    parser.add_argument(
        "--config-file",
        type=str,
        default="prod/config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output."
    )
    args = parser.parse_args()

    verbose = args.verbose

    # ----------------------
    # 2) Load & Validate Config
    # ----------------------
    config_manager = ConfigManager(args.config_file, verbose)
    config_manager.validate()
    config = config_manager.config

    if "data_config" not in config:
        raise ValueError("Configuration must include 'data_config' with 'n_num_features' and 'cat_cols'.")

    data_config = config["data_config"]
    required_keys = ["n_num_features", "cat_cols"]
    if not all(key in data_config for key in required_keys):
        raise ValueError(f"'data_config' must contain {required_keys}.")

    # ----------------------
    # 3) Set Up DatabaseManager (Optional)
    # ----------------------
    db_manager = None
    if "db_url" in config:
        db_manager = DatabaseManager(config["db_url"], verbose)

    # ----------------------
    # 4) Load Input Data
    # ----------------------
    if db_manager and "db_input_table" in config:
        # Load from Database
        input_data = db_manager.load_data(config["db_input_table"])
    elif "input_file" in config:
        # Load from CSV file
        if verbose:
            print(f"Loading input data from file '{config['input_file']}'...")
        try:
            input_data = pd.read_csv(config["input_file"])
            # Drop any unneeded columns
            input_data = input_data.drop(columns=["Unnamed: 0", "Tempo di riduzione diacetile"], errors='ignore')
            if verbose:
                print("Input data successfully loaded from CSV.")
        except FileNotFoundError:
            raise ValueError(f"Input file not found: {config['input_file']}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file '{config['input_file']}': {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error while loading input file: {e}")
    else:
        raise ValueError(
            "Configuration must include either 'input_file' or both 'db_url' and 'db_input_table'."
        )

    # ----------------------
    # 5) Initialize Device (CPU/CUDA)
    # ----------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ----------------------
    # 6) Initialize TabularPredictor
    # ----------------------
    if verbose:
        print(f"Initializing TabularPredictor with model '{config['model_name']}'...")

    try:
        tab_predictor = TabularPredictor(
            model_name=config["model_name"],
            path_to_model=config["model_file"],
            data_config=data_config,
            device=device
        )
        if verbose:
            print("TabularPredictor successfully initialized.")
    except Exception as e:
        raise ValueError(f"Failed to initialize TabularPredictor: {e}")

    # ----------------------
    # 7) Generate Predictions
    # ----------------------
    if verbose:
        print("Running predictions on the input data...")

    try:
        predictions = tab_predictor(input_data)
        if verbose:
            print("Predictions successfully generated.")
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")

    predictions_df = pd.DataFrame(predictions)

    # ----------------------
    # 8) Save Predictions
    # ----------------------
    if db_manager and "db_output_table" in config:
        # Save to Database
        db_manager.save_predictions_append(predictions_df, config["db_output_table"])
    elif "output_file" in config:
        # Save to CSV
        if verbose:
            print(f"Saving predictions to file '{config['output_file']}'...")
        try:
            predictions_df.to_csv(config["output_file"], index=False)
            if verbose:
                print("Predictions successfully saved to file.")
        except Exception as e:
            raise ValueError(f"Failed to save predictions to CSV: {e}")
    else:
        raise ValueError(
            "Configuration must include either 'output_file' or both 'db_url' and 'db_output_table'."
        )


if __name__ == "__main__":
    main()

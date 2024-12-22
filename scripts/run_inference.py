import argparse
import sys

import pandas as pd
import torch

sys.path.append('.')

from src.classes.periodicity.prod.TabularPredictor import TabularPredictor


def main():
    # Parse command-line arguments for model name, model file path, and input/output file path
    # Supported models: fnet, cnet, tabfnet, tabcnet, autopnpnet, tabautopnpnet
    parser = argparse.ArgumentParser(description="Run tabular predictions with a specified model.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output-file", type=str, default="predictions.csv", help="Path to the output CSV file.")
    parser.add_argument("--model-name", type=str, default="tabautopnpnet", help="Name of the model to load.")
    parser.add_argument("--model-file", type=str, default="models/tabautopnpnet.pth",
                        help="Path to the model .pth file.")

    args = parser.parse_args()

    # Load the input data from the provided CSV file
    try:
        input_data = pd.read_csv(args.input_file)
        input_data = input_data.drop(columns=["Unnamed: 0", "Tempo di riduzione diacetile"], errors='ignore')
    except Exception as e:
        # Raise a ValueError if the input file cannot be loaded
        raise ValueError(f"Failed to load input file: {e}")

    # Define the data configuration dictionary. This includes specifying:
    # - The number of numerical features (n_num_features)
    # - The names of the categorical columns (cat_cols)
    data_config = {
        "n_num_features": 23,
        "cat_cols": ["Brand"]
    }

    # Determine the device on which predictions will be made (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Instantiate the TabularPredictor with the provided model name, model file, and data configuration
    tab_predictor = TabularPredictor(args.model_name, args.model_file, data_config, device)

    # Run predictions on the input data using the loaded model
    predictions = tab_predictor(input_data)

    # Save the predictions to a CSV file
    pd.DataFrame(predictions).to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()

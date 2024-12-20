import argparse

import pandas as pd
import torch

from src.classes.periodicity.prod.TabularPredictor import TabularPredictor
from src.classes.utils.Logger import Logger

logger = Logger()


def main():
    """
    Main function to:
    - Parse command-line arguments
    - Load a model
    - Load and preprocess input data
    - Run prediction
    """

    parser = argparse.ArgumentParser(description="Run tabular predictions with a specified model.")
    parser.add_argument("--model-name", type=str, default="tabautopnpnet", help="Name of the model to load.")
    parser.add_argument("--model-file", type=str, default="models/tabautopnpnet.pth",
                        help="Path to the model .pth file.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input CSV file.")

    args = parser.parse_args()

    # Load the input data
    try:
        input_data = pd.read_csv(args.input_file)
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return

    data_config = {
        "n_num_features": 26,
        "cat_cols": ["Brand"]
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tab_predictor = TabularPredictor(args.model_name, args.model_file, data_config, device)

    # Run predictions
    predictions = tab_predictor(input_data)

    print(predictions)


if __name__ == "__main__":
    main()

import os
import sys
import tempfile
import copy
import logging
from typing import List, Dict, Tuple, Optional

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================================================
# --- Path Correction and Constants ---
# ==================================================================================
def get_base_path():
    """Gets the base path, whether running from a script or a frozen exe."""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        try:
            return os.path.dirname(os.path.abspath(__file__))
        except NameError:
            return os.path.abspath(".")

BASE_PATH = get_base_path()
SRC_PATH = os.path.join(BASE_PATH, 'src')
MODELS_DIR = os.path.join(BASE_PATH, 'models')
DEFAULT_CONFIG_PATH = os.path.join(BASE_PATH, 'prod', 'config.yaml')
DEFAULT_PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'preprocessor.joblib')

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from classes.periodicity.prod.TabularPredictor import TabularPredictor
from classes.utils.ConfigManager import ConfigManager

# This list is needed by both frontend and backend
MANUAL_INPUT_GROUPS = [
    {"group_name": "Gravity & Alcohol Metrics", "open": True, "fields": [
        {"label": "Grado primitivo %Pp", "key": "Grado primitivo %Pp", "type": "number", "value": 12.5},
        {"label": "Alcol vol %", "key": "Alcol_vol", "type": "number", "value": 5.0},
        {"label": "Alcol peso %", "key": "Alcol_peso", "type": "number", "value": 4.0},
        {"label": "ABD", "key": "ABD", "type": "number", "value": 3.5},
        {"label": "Estratto reale", "key": "Estratto reale", "type": "number", "value": 4.5},
        {"label": "Estratto apparente", "key": "Estratto apparente", "type": "number", "value": 2.5},
        {"label": "Estratto apparente limite %Pp", "key": "Estratto apparente limite %Pp", "type": "number",
         "value": 2.0}]},
    {"group_name": "Fermentation & Yeast Parameters", "fields": [
        {"label": "Temperatura Fermentazione (°C)", "key": "Temperatura Fermentazione", "type": "slider", "minimum": 8,
         "maximum": 25, "value": 18},
        {"label": "Temp. lievito all'insemenzamento (°C)", "key": "Temp. lievito all'insemenzamento", "type": "number",
         "value": 15},
        {"label": "Cellule all'insemenzamento (M/ml)", "key": "Cellule all'insemenzamento", "type": "number",
         "value": 10},
        {"label": "Durata di conservazione lievito in cella (days)", "key": "Durata di conservazione lievito in cella",
         "type": "number", "value": 7}]},
    {"group_name": "Process & Attenuation Metrics", "fields": [
        {"label": "Attenuazione Reale (RDF) %", "key": "Attenuazione Reale (RDF) %", "type": "number", "value": 75.0},
        {"label": "Attenuazione vendita apparente %p", "key": "Attenuazione vendita apparente %p", "type": "number",
         "value": 80.0},
        {"label": "Differenza apparente-limite", "key": "Differenza apparente-limite", "type": "number", "value": 0.5},
        {"label": "Fermentation rate (13.5°-5.5°)", "key": "Fermentation rate (13.5°-5.5°)", "type": "number",
         "value": 48},
        {"label": "Rapporto ER/ABW", "key": "Rapporto ER/ABW", "type": "number", "value": 1.1}]},
    {"group_name": "Chemical & Flavor Analysis", "fields": [
        {"label": "pH", "key": "pH", "type": "number", "value": 4.2},
        {"label": "Diacetile + precursori (ferm.)", "key": "Diacetile + precursori (ferm.)", "type": "number",
         "value": 50},
        {"label": "Pentandione + precursori", "key": "Pentandione + precursori", "type": "number", "value": 10},
        {"label": "NBB-A", "key": "NBB-A", "type": "number", "value": 1.0},
        {"label": "NBB-B", "key": "NBB-B", "type": "number", "value": 1.0},
        {"label": "Hopped Wort", "key": "Hopped Wort", "type": "number", "value": 100},
        {"label": "Hopped Wort (37°C)", "key": "Hopped Wort (37°C)", "type": "number", "value": 100}]}
]
EXPECTED_COLUMNS = [field['key'] for group in MANUAL_INPUT_GROUPS for field in group['fields']]


# ==================================================================================
# --- Core Logic Functions ---
# ==================================================================================

def find_models_hierarchically(base_dir: str) -> Dict[str, List[str]]:
    """Scans subdirectories to find models for each target variable."""
    model_dict = {}
    if not os.path.isdir(base_dir): return model_dict
    for target_name in os.listdir(base_dir):
        target_path = os.path.join(base_dir, target_name)
        if os.path.isdir(target_path):
            models = sorted([f for f in os.listdir(target_path) if f.endswith(('.pth', '.pt'))])
            if models: model_dict[target_name] = models
    return model_dict


def get_config(path: str):
    """Loads and validates the config."""
    logging.info(f"Loading configuration from {path}.")
    config_manager = ConfigManager(path)
    config_manager.validate()
    logging.info("Configuration loaded and validated successfully.")
    return config_manager.config


def prepare_input_dataframe(input_method: str, manual_inputs: List,
                            batch_file: Optional[tempfile.TemporaryFile]) -> pd.DataFrame:
    """Prepares the input DataFrame from either the manual form or a CSV file."""
    logging.info(f"Preparing input DataFrame using method: {input_method}")
    if input_method == "Manual Entry":
        if any(i is None or i == '' for i in manual_inputs):
            raise ValueError("Please fill in all manual input fields.")
        input_dict = {key: [value] for key, value in zip(EXPECTED_COLUMNS, manual_inputs)}
        df = pd.DataFrame(input_dict)
        logging.info(f"Created DataFrame from manual inputs:\n{df.to_string()}")
        return df
    elif input_method == "Batch from CSV":
        if batch_file is None:
            raise ValueError("Please upload a CSV file for batch prediction.")
        temp_df = pd.read_csv(batch_file.name)
        missing_cols = set(EXPECTED_COLUMNS) - set(temp_df.columns)
        if missing_cols:
            raise ValueError(f"CSV is missing required columns: {missing_cols}")
        logging.info(f"Loaded DataFrame from CSV with shape {temp_df.shape}.")
        return temp_df[EXPECTED_COLUMNS]
    else:
        raise ValueError("Invalid input method.")


def run_single_prediction(df: pd.DataFrame, target_variable: str, model_filename: str) -> Tuple:
    """Handles prediction for a single selected model."""
    logging.info(f"Running single prediction for target '{target_variable}' with model '{model_filename}'.")
    config = get_config(DEFAULT_CONFIG_PATH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(MODELS_DIR, target_variable, model_filename)
    model_name_key = os.path.splitext(model_filename)[0]

    data_config_copy = copy.deepcopy(config["data_config"])
    logging.info("Created a deepcopy of the data configuration.")

    predictor = TabularPredictor(model_name=model_name_key,
                                 path_to_model=model_path,
                                 path_to_preprocessor=DEFAULT_PREPROCESSOR_PATH,
                                 data_config=data_config_copy,
                                 device=device)
    logging.info("TabularPredictor initialized.")

    predictions = predictor(df)
    logging.info(f"Raw predictions from model: {predictions}")

    if df.shape[0] == 1:
        scalar_prediction = predictions.item()
        logging.info(f"Final scalar prediction: {scalar_prediction:.3f}")
        return f"{scalar_prediction:.2f}", None
    else:
        output_df = df.copy()
        output_df[f"prediction_{model_name_key}"] = predictions
        logging.info("Appended predictions to batch DataFrame.")
        return None, output_df


def run_multi_model_comparison(df: pd.DataFrame, target_variable: str, selected_models: List[str]):
    """Handles prediction and comparison for multiple selected models."""
    logging.info(f"Running multi-model comparison for target '{target_variable}'.")
    config = get_config(DEFAULT_CONFIG_PATH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_predictions = [df.copy()]

    for model_filename in selected_models:
        logging.info(f"Processing model: {model_filename}")
        model_path = os.path.join(MODELS_DIR, target_variable, model_filename)
        model_name_key = os.path.splitext(model_filename)[0]
        try:
            data_config_copy = copy.deepcopy(config["data_config"])
            predictor = TabularPredictor(model_name=model_name_key,
                                         path_to_model=model_path,
                                         path_to_preprocessor=DEFAULT_PREPROCESSOR_PATH,
                                         data_config=data_config_copy,
                                         device=device)
            predictions = predictor(df)
            logging.info(f"Raw predictions from '{model_filename}': {predictions}")
            all_predictions.append(pd.DataFrame(predictions, columns=[f"prediction_{model_name_key}"], index=df.index))
        except Exception as e:
            logging.error(f"Failed to process model '{model_filename}': {e}", exc_info=True)
            # In a real app, you might want to raise this or handle it more gracefully
            raise e

    if len(all_predictions) <= 1: raise ValueError("No models were successfully processed.")

    final_df = pd.concat(all_predictions, axis=1)
    pred_cols = [col for col in final_df.columns if col.startswith('prediction_')]
    summary_stats = final_df[pred_cols].describe().reset_index()

    plt.style.use('seaborn-v0_8-whitegrid')
    dist_fig, ax = plt.subplots(figsize=(8, 6))
    for col in pred_cols: sns.kdeplot(final_df[col], ax=ax, label=col, fill=True, alpha=0.3)
    ax.set_title("Prediction Distributions Comparison", fontsize=14)
    ax.legend()
    plt.tight_layout()
    logging.info("Multi-model comparison finished successfully.")
    return final_df, summary_stats, dist_fig

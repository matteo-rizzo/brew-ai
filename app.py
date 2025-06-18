import os
import sys
import tempfile
from functools import lru_cache
from typing import List, Dict, Tuple, Optional

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib


# ==================================================================================
# --- Path Correction for PyInstaller ---
# This block of code is essential for the frozen executable to find your data files.
# ==================================================================================
def get_base_path():
    """Gets the base path, whether running from a script or a frozen exe."""
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the base path is the temp folder where PyInstaller extracts everything.
        return sys._MEIPASS
    else:
        # If run as a normal script, the base path is the script's directory.
        return os.path.dirname(os.path.abspath(__file__))


BASE_PATH = get_base_path()
SRC_PATH = os.path.join(BASE_PATH, 'src')
MODELS_DIR = os.path.join(BASE_PATH, 'models')
DEFAULT_CONFIG_PATH = os.path.join(BASE_PATH, 'prod', 'config.yaml')

# Now, we add the corrected src path to sys.path
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from classes.periodicity.prod.TabularPredictor import TabularPredictor
from classes.utils.ConfigManager import ConfigManager

# ==================================================================================
# --- USER CONFIGURATION ---
# ==================================================================================
SUPPORTED_TARGETS = ["Diacetyl Rest", "Final Gravity", "Ester Production (ppm)"]
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
        # BUG FIX: Changed 'min' and 'max' to 'minimum' and 'maximum' for the gr.Slider component.
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
# --- Core Logic - Refactored for Clarity & Performance ---
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


@lru_cache(maxsize=4)
def get_config(path: str):
    """Loads and validates the config, caching the result."""
    gr.Info(f"Loading configuration from {path}...")
    config_manager = ConfigManager(path)
    config_manager.validate()
    return config_manager.config


def prepare_input_dataframe(input_method: str, manual_inputs: List,
                            batch_file: Optional[tempfile.TemporaryFile]) -> pd.DataFrame:
    """Prepares the input DataFrame from either the manual form or a CSV file."""
    if input_method == "Manual Entry":
        if any(i is None or i == '' for i in manual_inputs): raise ValueError("Please fill in all manual input fields.")
        input_dict = {key: [value] for key, value in zip(EXPECTED_COLUMNS, manual_inputs)}
        return pd.DataFrame(input_dict)
    elif input_method == "Batch from CSV":
        if batch_file is None: raise ValueError("Please upload a CSV file for batch prediction.")
        temp_df = pd.read_csv(batch_file.name)
        missing_cols = set(EXPECTED_COLUMNS) - set(temp_df.columns)
        if missing_cols: raise ValueError(f"CSV is missing required columns: {missing_cols}")
        return temp_df[EXPECTED_COLUMNS]
    else:
        raise ValueError("Invalid input method.")


def run_single_prediction(df: pd.DataFrame, target_variable: str, model_filename: str) -> Tuple:
    """Handles prediction for a single selected model."""
    config = get_config(DEFAULT_CONFIG_PATH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(MODELS_DIR, target_variable, model_filename)
    model_name_key = os.path.splitext(model_filename)[0]

    predictor = TabularPredictor(model_name=model_name_key, path_to_model=model_path, data_config=config["data_config"],
                                 device=device)
    predictions = predictor(df)

    if df.shape[0] == 1:  # Manual entry
        scalar_prediction = predictions.item()
        return f"{scalar_prediction:.3f}", None
    else:  # Batch entry
        output_df = df.copy()
        output_df[f"prediction_{model_name_key}"] = predictions
        return None, output_df


def run_multi_model_comparison(df: pd.DataFrame, target_variable: str, selected_models: List[str],
                               progress=gr.Progress(track_tqdm=True)):
    """Handles prediction and comparison for multiple selected models."""
    config = get_config(DEFAULT_CONFIG_PATH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_predictions = [df.copy()]

    for model_filename in progress.tqdm(selected_models, desc="Processing Models"):
        model_path = os.path.join(MODELS_DIR, target_variable, model_filename)
        model_name_key = os.path.splitext(model_filename)[0]
        try:
            predictor = TabularPredictor(model_name=model_name_key, path_to_model=model_path,
                                         data_config=config["data_config"], device=device)
            predictions = predictor(df)
            all_predictions.append(pd.DataFrame(predictions, columns=[f"prediction_{model_name_key}"], index=df.index))
        except Exception as e:
            gr.Warning(f"Failed to process model '{model_filename}': {e}. Skipping.")

    if len(all_predictions) <= 1: raise ValueError("No models were successfully processed.")

    final_df = pd.concat(all_predictions, axis=1)
    pred_cols = [col for col in final_df.columns if col.startswith('prediction_')]
    summary_stats = final_df[pred_cols].describe().reset_index()

    dist_fig, ax = plt.subplots(figsize=(8, 6));
    plt.style.use('seaborn-v0_8-whitegrid')
    for col in pred_cols: sns.kdeplot(final_df[col], ax=ax, label=col, fill=True, alpha=0.3)
    ax.set_title("Prediction Distributions Comparison", fontsize=14);
    ax.legend();
    plt.tight_layout()

    return final_df, summary_stats, dist_fig


def master_prediction_handler(*args):
    """
    Main dispatcher function. It validates inputs and routes to the correct prediction handler.
    *args contains all Gradio inputs in order.
    """
    try:
        input_method, target_variable, selected_models, batch_file = args[0], args[1], args[2], args[3]
        manual_inputs = args[4:]

        if not target_variable: raise ValueError("Please select a target variable.")
        if not selected_models: raise ValueError("No models selected.")

        df = prepare_input_dataframe(input_method, manual_inputs, batch_file)
        is_comparison = len(selected_models) > 1

        if not is_comparison:
            single_pred, batch_pred = run_single_prediction(df, target_variable, selected_models[0])
            gr.Info("Prediction successful!")
            return single_pred, batch_pred, None, None, gr.update(visible=True), gr.update(visible=False)
        else:
            comparison_df, summary_stats, dist_plot = run_multi_model_comparison(df, target_variable, selected_models)
            gr.Info("Comparison successful!")
            return None, comparison_df, summary_stats, dist_plot, gr.update(visible=False), gr.update(visible=True)

    except Exception as e:
        gr.Error(f"An error occurred: {e}");
        print(f"ERROR: {e}")
        return None, None, None, None, gr.update(visible=True), gr.update(visible=False)


# ==================================================================================
# --- UI Helper Functions ---
# ==================================================================================
def create_manual_form() -> List[gr.components.Component]:
    """Dynamically creates the manual input form UI from the configuration."""
    components = []
    for group in MANUAL_INPUT_GROUPS:
        with gr.Accordion(group['group_name'], open=group.get('open', False)):
            with gr.Row():
                for field in group['fields']:
                    with gr.Column():
                        comp_map = {
                            'number': gr.Number,
                            'slider': gr.Slider,
                            'dropdown': gr.Dropdown
                        }
                        comp_class = comp_map.get(field['type'], gr.Textbox)
                        # This dictionary unpacking requires correct argument names
                        comp_args = {k: v for k, v in field.items() if k not in ['key', 'type']}
                        components.append(comp_class(**comp_args))
    return components


def clear_interface():
    """Resets all input and output components to their initial state."""
    flat_fields = [field for group in MANUAL_INPUT_GROUPS for field in group['fields']]
    default_form_values = [field.get('value') for field in flat_fields]
    updates = [
        gr.update(value=None), gr.update(value=None, choices=[], interactive=False),
        gr.update(value="Manual Entry"), gr.update(value=None),
        gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None),
        gr.update(visible=True), gr.update(visible=False)
    ]
    updates.extend(default_form_values)
    return updates


# ==================================================================================
# --- Gradio UI Definition ---
# ==================================================================================
AVAILABLE_MODELS_DICT = find_models_hierarchically(MODELS_DIR)

with gr.Blocks(theme=gr.themes.Soft(primary_hue="yellow"),
               css=".gradio-container {max-width: 95% !important;}") as demo:
    gr.Markdown("# 🍺 Fermentation Process Prediction Engine")
    gr.Markdown(
        "A SMACT competence centre project by Matteo Rizzo, in collaboration with Ca' Foscari University of Venice and Omnia Technologies.")

    with gr.Row():
        with gr.Column(scale=2, min_width=400):
            gr.Markdown("## ⚙️ Configuration")
            with gr.Group():
                gr.Markdown("### Step 1: Select Target & Model(s)")
                target_variable_dd = gr.Dropdown(choices=list(AVAILABLE_MODELS_DICT.keys()),
                                                 label="Select Target Variable")
                model_selection_group = gr.CheckboxGroup(label="Select Model(s)",
                                                         info="Choose one or more models to run.", interactive=False)

            with gr.Group():
                gr.Markdown("### Step 2: Provide Input Data")
                input_method_radio = gr.Radio(choices=["Manual Entry", "Batch from CSV"], value="Manual Entry",
                                              label="Input Method")
                with gr.Group(visible=True) as manual_form_group:
                    manual_inputs_list = create_manual_form()
                with gr.Group(visible=False) as batch_file_group:
                    batch_file_input = gr.File(label="Upload CSV File", file_types=[".csv"])

            with gr.Group():
                gr.Markdown("### Step 3: Run")
                run_button = gr.Button("🚀 Run Prediction / Comparison", variant="primary", interactive=False)
                clear_button = gr.Button("🔄 Clear All")

        with gr.Column(scale=3):
            gr.Markdown("## 📈 Results")
            with gr.Group(visible=True) as single_result_group:
                gr.Markdown("#### Single Prediction Result")
                single_prediction_output = gr.Textbox(label="Predicted Value", interactive=False)
            with gr.Group(visible=False) as comparison_result_group:
                gr.Markdown("#### Batch / Comparison Results")
                with gr.Tabs():
                    with gr.TabItem("Side-by-Side Data"):
                        comparison_df_output = gr.DataFrame(label="Predictions with Input Data", wrap=True)
                    with gr.TabItem("Statistics"):
                        summary_stats_output = gr.DataFrame(label="Descriptive Statistics of Predictions")
                    with gr.TabItem("Distributions Plot"):
                        dist_plot_output = gr.Plot()

    # --- UI Interactivity Logic ---
    data_loaded_state = gr.State(False)


    def update_model_selection(target_variable: str):
        models = AVAILABLE_MODELS_DICT.get(target_variable, [])
        return gr.CheckboxGroup(choices=models, label="Select Model(s)", interactive=True,
                                value=[models[0]] if models else None)


    target_variable_dd.change(fn=update_model_selection, inputs=target_variable_dd, outputs=model_selection_group)


    def toggle_input_method_visibility(method: str):
        is_manual = method == "Manual Entry"
        return gr.update(visible=is_manual), gr.update(visible=not is_manual), True


    data_input_change_outputs = [manual_form_group, batch_file_group, data_loaded_state]
    input_method_radio.change(fn=toggle_input_method_visibility, inputs=input_method_radio,
                              outputs=data_input_change_outputs)
    batch_file_input.upload(fn=lambda: True, outputs=data_loaded_state)


    def toggle_run_button_visibility(data_loaded: bool, models_selected: List):
        return gr.Button(interactive=bool(data_loaded and models_selected))


    data_loaded_state.change(fn=toggle_run_button_visibility, inputs=[data_loaded_state, model_selection_group],
                             outputs=run_button)
    model_selection_group.change(fn=toggle_run_button_visibility, inputs=[data_loaded_state, model_selection_group],
                                 outputs=run_button)

    prediction_inputs = [input_method_radio, target_variable_dd, model_selection_group,
                         batch_file_input] + manual_inputs_list
    prediction_outputs = [single_prediction_output, comparison_df_output, summary_stats_output, dist_plot_output,
                          single_result_group, comparison_result_group]
    run_button.click(fn=master_prediction_handler, inputs=prediction_inputs, outputs=prediction_outputs,
                     show_progress="full")

    clear_button_outputs = [
                               target_variable_dd, model_selection_group, input_method_radio, batch_file_input,
                               single_prediction_output, comparison_df_output, summary_stats_output, dist_plot_output,
                               single_result_group, comparison_result_group
                           ] + manual_inputs_list
    clear_button.click(fn=clear_interface, outputs=clear_button_outputs)

if __name__ == "__main__":
    if not AVAILABLE_MODELS_DICT:
        print("\n[WARNING] The 'models' directory is empty or contains no valid subdirectories for targets.")
        print(
            "Please create subdirectories named after your targets (e.g., 'Diacetyl Rest/') and place model files inside.\n")
    demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False, inbrowser=True)

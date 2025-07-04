import logging
import signal
import sys
from typing import List

import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns

# Import the backend logic
import backend

# ==================================================================================
# --- Logging Setup ---
# ==================================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)


# ==================================================================================
# --- Main Application Logic (Handler) ---
# ==================================================================================

def master_prediction_handler(*args):
    """
    Main dispatcher function. It validates inputs and routes to the correct prediction handler
    from the backend.
    """
    logging.info("=" * 50)
    logging.info("MASTER PREDICTION HANDLER TRIGGERED")
    logging.info("=" * 50)

    try:
        # Unpack Gradio inputs
        input_method, target_variable, selected_models, batch_file = args[0], args[1], args[2], args[3]
        manual_inputs = args[4:]

        logging.info(
            f"Received inputs: method='{input_method}', target='{target_variable}', models='{selected_models}'")
        logging.info(f"Received manual inputs (first 5): {manual_inputs[:5]}")

        # Input validation
        if not target_variable: raise ValueError("Please select a target variable.")
        if not selected_models: raise ValueError("No models selected.")

        # Delegate data preparation to the backend
        df = backend.prepare_input_dataframe(input_method, manual_inputs, batch_file)

        # Determine which view to show.
        is_single_manual_prediction = df.shape[0] == 1 and len(selected_models) == 1

        if is_single_manual_prediction:
            # This is the only case where we show the single prediction output text box.
            single_pred, _ = backend.run_single_prediction(df, target_variable, selected_models[0])
            gr.Info("✅ Prediction successful!")
            return single_pred, None, None, None, gr.update(visible=True), gr.update(visible=False)
        else:
            # All other cases (batch input OR multi-model comparison) show the comparison view.
            if len(selected_models) == 1:
                # This handles a batch prediction with a single model.
                _, comparison_df = backend.run_single_prediction(df, target_variable, selected_models[0])

                # Generate stats and plot for the single model's batch predictions.
                prediction_col_name = [col for col in comparison_df.columns if col.startswith('prediction_')][-1]
                summary_stats = comparison_df[[prediction_col_name]].describe().reset_index()

                plt.style.use('seaborn-v0_8-whitegrid')
                dist_fig, ax = plt.subplots(figsize=(8, 6))
                sns.kdeplot(comparison_df[prediction_col_name], ax=ax, label=prediction_col_name, fill=True, alpha=0.5)
                ax.set_title(f"Distribution for {prediction_col_name}", fontsize=14)
                ax.legend()
                plt.tight_layout()
                dist_plot = dist_fig

                gr.Info("✅ Batch prediction successful!")
            else:
                # This handles any multi-model comparison.
                comparison_df, summary_stats, dist_plot = backend.run_multi_model_comparison(df, target_variable,
                                                                                             selected_models)
                gr.Info("✅ Comparison successful!")

            def style_dataframe(df_to_style, highlight=False):
                if df_to_style is None:
                    return None

                # Select only numeric columns for float formatting
                numeric_cols = df_to_style.select_dtypes(include=['number']).columns
                formatter = {col: '{:.2f}' for col in numeric_cols}

                # Apply formatting and highlighting using the Styler API
                styler = df_to_style.style.format(formatter, na_rep="-")

                # Identify prediction columns for highlighting
                pred_cols = [col for col in df_to_style.columns if 'prediction' in str(col)]

                if highlight and pred_cols:
                    # FIX: Use set_properties for a cleaner and more robust way to apply styles to columns
                    styler = styler.set_properties(
                        subset=pred_cols,
                        **{'background-color': '#FFFBEB', 'font-weight': 'bold'}
                    )

                return styler

            # Apply styling to both dataframes
            comparison_df = style_dataframe(comparison_df, highlight=True)
            summary_stats = style_dataframe(summary_stats)

            # Return the results to the appropriate components in the comparison view.
            return None, comparison_df, summary_stats, dist_plot, gr.update(visible=False), gr.update(visible=True)

    except Exception as e:
        logging.error(f"An error occurred in master_prediction_handler: {e}", exc_info=True)
        gr.Error(f"An error occurred: {e}")
        return None, None, None, None, gr.update(visible=True), gr.update(visible=False)


# ==================================================================================
# --- UI Helper Functions ---
# ==================================================================================
def create_manual_form() -> List[gr.components.Component]:
    """Dynamically creates the manual input form UI from the configuration."""
    components = []
    for group in backend.MANUAL_INPUT_GROUPS:
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
                        comp_args = {k: v for k, v in field.items() if k not in ['key', 'type']}
                        components.append(comp_class(**comp_args))
    return components


def clear_interface():
    """Resets all input and output components to their initial state."""
    logging.info("Clearing the interface.")
    flat_fields = [field for group in backend.MANUAL_INPUT_GROUPS for field in group['fields']]
    default_form_values = [field.get('value') for field in flat_fields]

    updates = [
        gr.update(value=None),
        gr.update(value=None, choices=[], interactive=False),
        gr.update(value="Manual Entry"),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(visible=True),
        gr.update(visible=False),
    ]
    updates.extend(default_form_values)
    return updates


# ==================================================================================
# --- Gradio UI Definition ---
# ==================================================================================
def build_ui():
    """Builds and returns the Gradio Blocks UI."""
    AVAILABLE_MODELS_DICT = backend.find_models_hierarchically(backend.MODELS_DIR)

    # Custom CSS for a more polished and modern look
    custom_css = """
    /* General container styling */
    .gradio-container { font-family: 'Inter', sans-serif; }
    /* Custom styling for cards/groups */
    .gradio-group {
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        border-radius: 12px !important;
        background: white;
        padding: 1rem;
    }
    /* Add padding to group titles */
    .group-title h3 {
        padding: 0.5rem;
        margin: 0 !important;
    }
    /* Custom button styling */
    .gradio-button {
        border-radius: 8px !important;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
        transition: all 0.2s ease-in-out;
    }
    .gradio-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    }
    /* Primary button styling */
    .primary {
        background: linear-gradient(to right, #fbbf24, #f59e0b) !important;
        color: white !important;
        border: none !important;
    }
    """

    # Using a base theme for easier customization
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        # Custom properties for the theme
        body_background_fill="#F9FAFB",
        block_background_fill="white",
        block_border_width="0",
        block_shadow="0 1px 3px 0 rgba(0, 0, 0, 0.05)",
        block_radius="12px",
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
        button_large_padding="12px 24px",
    )

    with gr.Blocks(theme=theme, css=custom_css) as demo:
        # --- Header ---
        with gr.Row():
            gr.Markdown(
                """
                <div style="text-align: center; width: 100%; padding: 1rem 0;">
                    <h1 style="color: #1F2937; font-size: 2.5rem; font-weight: 700;">🍺 Fermentation Process Prediction Engine</h1>
                </div>
                """
            )

        # --- Main Content ---
        with gr.Row():
            # --- Left Column: Configuration ---
            with gr.Column(scale=2, min_width=450):
                with gr.Column():
                    with gr.Group():
                        gr.Markdown("### 1. Select Target & Model(s)", elem_classes="group-title")
                        target_variable_dd = gr.Dropdown(choices=list(AVAILABLE_MODELS_DICT.keys()),
                                                         label="Target Variable")
                        model_selection_group = gr.CheckboxGroup(label="Model(s) to Run",
                                                                 info="Choose one for a single prediction or multiple for a comparison.",
                                                                 interactive=False)

                    with gr.Group():
                        gr.Markdown("### 2. Provide Input Data", elem_classes="group-title")
                        input_method_radio = gr.Radio(choices=["Manual Entry", "Batch from CSV"], value="Manual Entry",
                                                      label="Input Method")
                        with gr.Column(visible=True) as manual_form_group:
                            manual_inputs_list = create_manual_form()
                        with gr.Column(visible=False) as batch_file_group:
                            batch_file_input = gr.File(label="Upload CSV File", file_types=[".csv"])

                    gr.Markdown("")  # Spacer

                    with gr.Row():
                        quit_button = gr.Button("Quit Application", variant="stop")
                        clear_button = gr.Button("Clear All", variant="secondary")
                        run_button = gr.Button("Run Prediction", variant="primary", interactive=False)

            # --- Right Column: Results ---
            with gr.Column(scale=3, min_width=600):
                with gr.Column(visible=True) as single_result_group:
                    single_prediction_output = gr.Textbox(label="Predicted Value", interactive=False,
                                                          elem_id="single-prediction-output")
                with gr.Column(visible=False) as comparison_result_group:
                    with gr.Tabs():
                        with gr.TabItem("Side-by-Side Data"):
                            comparison_df_output = gr.DataFrame(label="Predictions with Input Data", wrap=True,
                                                                interactive=False)
                        with gr.TabItem("Statistics"):
                            summary_stats_output = gr.DataFrame(label="Descriptive Statistics of Predictions",
                                                                interactive=False)
                        with gr.TabItem("Distributions Plot"):
                            dist_plot_output = gr.Plot()

        # --- Footer ---
        with gr.Row():
            gr.HTML(
                """
                <div style="display: flex; align-items: center; justify-content: center; width: 100%; padding: 1.5rem 0; border-top: 1px solid #E5E7EB; margin-top: 2rem; background-color: #FFFFFF;">
                    <div style="display: flex; align-items: center; gap: 2rem; margin-right: 2rem;">
                        <img src="gradio_api/file=assets/unive-logo.png" alt="Ca' Foscari University Logo" style="height: 50px;">
                        <img src="gradio_api/file=assets/smact-logo.png" alt="SMACT Logo" style="height: 50px;">
                        <img src="gradio_api/file=assets/omnia-technologies-logo.png" alt="Omnia Technologies Logo" style="height: 50px;">
                        <img src="gradio_api/file=assets/next-generation-eu-logo.png" alt="BRIDGE Project Logo" style="height: 50px;">
                    </div>
                    <div style="text-align: left; border-left: 1px solid #E5E7EB; padding-left: 2rem;">
                        <p style="margin: 0; font-weight: bold; color: #4B5563;">Progetto BRIDGE - Brewing Research within Intensive Data Gathering Environments</p>
                        <p style="margin: 0; font-size: 0.8rem; color: #6B7280;">CUP H39J23001590004 - COR 16298020 - Next Gen. Eu - M4C2I2.3</p>
                    </div>
                </div>
                """
            )

        # --- UI Interactivity Logic ---
        data_ready_state = gr.State(True)

        def shutdown_app():
            logging.info("Quit button clicked. Shutting down.")
            demo.close()

        quit_button.click(fn=shutdown_app, js="() => { window.close(); }")

        def update_model_selection(target_variable: str):
            models = AVAILABLE_MODELS_DICT.get(target_variable, [])
            return gr.CheckboxGroup(choices=models, label="Select Model(s)", interactive=True,
                                    value=[models[0]] if models else None)

        def initial_setup():
            """Pre-selects the first available target and model when the UI loads."""
            if AVAILABLE_MODELS_DICT:
                first_target = list(AVAILABLE_MODELS_DICT.keys())[0]
                models_for_first_target = AVAILABLE_MODELS_DICT.get(first_target, [])
                # Return the initial value for the dropdown and an update object for the checkbox group
                return first_target, gr.update(
                    choices=models_for_first_target,
                    value=[models_for_first_target[0]] if models_for_first_target else None,
                    interactive=True
                )
            # If no models are found, return default empty states
            return None, gr.update(choices=[], value=None, interactive=False)

        target_variable_dd.change(fn=update_model_selection, inputs=target_variable_dd, outputs=model_selection_group)

        def toggle_input_method(method: str):
            is_manual = method == "Manual Entry"
            return gr.update(visible=is_manual), gr.update(visible=not is_manual), is_manual

        input_method_radio.change(fn=toggle_input_method, inputs=input_method_radio,
                                  outputs=[manual_form_group, batch_file_group, data_ready_state])

        batch_file_input.upload(fn=lambda: True, outputs=data_ready_state)
        for component in manual_inputs_list:
            component.change(fn=lambda: True, outputs=data_ready_state)

        def toggle_run_button_interactivity(data_ready: bool, models_selected: List):
            is_interactive = bool(data_ready and models_selected)
            button_text = "Run Comparison" if len(models_selected or []) > 1 else "Run Prediction"
            return gr.Button(value=button_text, interactive=is_interactive)

        data_ready_state.change(fn=toggle_run_button_interactivity, inputs=[data_ready_state, model_selection_group],
                                outputs=run_button)
        model_selection_group.change(fn=toggle_run_button_interactivity,
                                     inputs=[data_ready_state, model_selection_group],
                                     outputs=run_button)

        prediction_inputs = [input_method_radio, target_variable_dd, model_selection_group,
                             batch_file_input] + manual_inputs_list
        prediction_outputs = [single_prediction_output, comparison_df_output, summary_stats_output, dist_plot_output,
                              single_result_group, comparison_result_group]

        run_button.click(fn=master_prediction_handler, inputs=prediction_inputs, outputs=prediction_outputs,
                         show_progress="full")

        clear_button_outputs = [target_variable_dd, model_selection_group, input_method_radio, batch_file_input,
                                single_prediction_output, comparison_df_output, summary_stats_output,
                                dist_plot_output, single_result_group, comparison_result_group] + manual_inputs_list

        clear_button.click(fn=clear_interface, outputs=clear_button_outputs)

        # Add the load event to initialize the UI state
        demo.load(fn=initial_setup, outputs=[target_variable_dd, model_selection_group])

    return demo


if __name__ == "__main__":
    app_ui = build_ui()


    # --- Graceful Shutdown Logic ---
    def shutdown_handler(signum, frame):
        logging.info("Shutdown signal received. Closing application.")
        app_ui.close()
        sys.exit(0)


    # Register the handler for SIGTERM (used by OS) and SIGINT (Ctrl+C)
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    # --- End Shutdown Logic ---

    app_ui.launch(server_name="127.0.0.1", server_port=7860, show_api=False, inbrowser=True, allowed_paths=["assets"])

import json
import subprocess

# Load dataset configurations from JSON
with open('benchmark/num_reg_config.json') as f:
    dataset_configs = json.load(f)

skipped_datasets = [
    "nyc-taxi-green-dec-2016",
    "delays_zurich_transport",
    "Allstate_Claims_Severity",
    "Airlines_DepDelay_1M"
]

# List of model types to iterate over
model_types = [
    "tabbaseline", "fnet", "tabfnet", "cnet", "tabcnet", "pnpnet", "tabpnpnet", "autopnpnet", "tabautopnpnet"
]

# Iterate over dataset IDs and model types
for dataset_id in dataset_configs.keys():
    for model_type in model_types:
        if dataset_id not in skipped_datasets:
            print(f"Evaluating model {model_type} on dataset {dataset_id}")
            # Run the evaluation command
            subprocess.run([
                "python", "src/evaluate_periodicity_models.py",
                "--model", model_type,
                "--dataset", dataset_id
            ])

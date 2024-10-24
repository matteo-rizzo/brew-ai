#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Export the PYTHONPATH to the current directory
export PYTHONPATH="."

# Parse the config.json and iterate over dataset IDs and models using Python
python - <<END
import json
import subprocess

# Load dataset configurations from JSON
with open('dataset/config.json') as f:
    dataset_configs = json.load(f)

# List of model types to iterate over
model_types = [
    "tabbaseline", "fnet", "tabfnet", "cnet",
    "tabcnet", "pnpnet", "tabpnpnet", "autopnpnet", "tabautopnpnet"
]

# Iterate over dataset IDs and model types
for dataset_id in dataset_configs.keys():
    for model_type in model_types:
        print(f"Evaluating model {model_type} on dataset {dataset_id}")
        # Run the evaluation command
        subprocess.run([
            "python", "src/evaluate_periodicity_models.py",
            "--model", model_type,
            "--dataset", dataset_id  # Assuming you need to pass the dataset ID
        ])
END

# Deactivate the virtual environment
deactivate

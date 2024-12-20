import json
import subprocess

# Load dataset configurations from JSON
with open('benchmark/num_clf_config.json') as f:
    dataset_configs = json.load(f)

skipped_datasets = [
    "nyc-taxi-green-dec-2016",
    "delays_zurich_transport",
    "Allstate_Claims_Severity",
    "Airlines_DepDelay_1M",
    "topo_2_1",
    "seattlecrime6",
    "particulate-matter-ukair-2017",
    "Mercedes_Benz_Greener_Manufacturing",
    "Higgs",
    "MiniBooNE",
    "covertype",
    "jannis",
    "road-safety"
]

# List of model types to iterate over
model_types = ["fnet", "opnet", "pnpnet", "autopnpnet"]

# Iterate over dataset IDs and model types
for dataset_id in dataset_configs.keys():
    for model_type in model_types:
        if dataset_id in skipped_datasets:
            print(f"Evaluating model {model_type} on dataset {dataset_id}")
            # Run the evaluation command
            subprocess.run([
                "python", "src/evaluate_periodicity_models.py",
                "--model", model_type,
                "--dataset", dataset_id
            ])

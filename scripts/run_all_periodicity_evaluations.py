import json
import subprocess

# List of model types to iterate over
model_types = ["fnet", "opnet", "autopnpnet", "tabbaseline"]

EVAL_BENCHMARKS = False
EVAL_DATASETS = True

if EVAL_BENCHMARKS:
    # Load benchmarks configurations from JSON
    with open('benchmark/num_clf_config.json') as f:
        benchmark_configs = json.load(f)

    skipped_benchmarks = [
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

    # Iterate over dataset IDs and model types
    for dataset_id in benchmark_configs.keys():
        for model_type in model_types:
            if dataset_id not in skipped_benchmarks:
                print(f"Evaluating model '{model_type}' on benchmark dataset: '{dataset_id}'")
                # Run the evaluation command
                subprocess.run([
                    "python", "scripts/evaluate_periodicity_models.py",
                    "--model", model_type,
                    "--dataset", dataset_id
                ])

if EVAL_DATASETS:
    # Load dataset configurations from JSON
    with open('dataset/config.json') as f:
        dataset_configs = json.load(f)

    skipped_datasets = []

    # Iterate over dataset IDs and model types
    for dataset_id in dataset_configs.keys():
        for model_type in model_types:
            if dataset_id not in skipped_datasets:
                print(f"Evaluating model {model_type} on dataset {dataset_id}")
                # Run the evaluation command
                subprocess.run([
                    "python", "scripts/evaluate_periodicity_models.py",
                    "--model", model_type,
                    "--dataset", dataset_id
                ])

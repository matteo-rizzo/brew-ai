#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Export the PYTHONPATH to the current directory
export PYTHONPATH="."

# Run the Python scripts
#python src/evaluate_deep_models.py --model_type "tabnet"
#python src/evaluate_deep_models.py --model_type "fttransformer"
#python src/evaluate_deep_models.py --model_type "tabtransformer"
python src/evaluate_periodicity_models.py --model_type "tabbaseline"
python src/evaluate_periodicity_models.py --model_type "fnet"
python src/evaluate_periodicity_models.py --model_type "tabfnet"
python src/evaluate_periodicity_models.py --model_type "cnet"
python src/evaluate_periodicity_models.py --model_type "tabcnet"
python src/evaluate_periodicity_models.py --model_type "pnpnet"
python src/evaluate_periodicity_models.py --model_type "tabpnpnet"
python src/evaluate_periodicity_models.py --model_type "autopnpnet"
python src/evaluate_periodicity_models.py --model_type "tabautopnpnet"
# python src/grid_search.py

# Deactivate the virtual environment
deactivate

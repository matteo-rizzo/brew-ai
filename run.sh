#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Export the PYTHONPATH to the current directory
export PYTHONPATH="."

# Run the Python scripts
python src/evaluate_deep_models.py --model-type "tabnet"
python src/evaluate_deep_models.py --model-type "fttransformer"
# python src/evaluate_deep_models.py --model-type "tabtransformer"
python src/grid_search.py

# Deactivate the virtual environment
deactivate

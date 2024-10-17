#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Run the Python scripts
python src/grid_search.py
python src/evaluate_deep_models.py

# Deactivate the virtual environment
deactivate

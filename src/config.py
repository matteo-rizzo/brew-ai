import numpy as np
import torch

# Common
BASE_LOG_DIR = "logs"
RANDOM_SEED = 0
NUM_FOLDS = 5
TEST_SIZE = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ID = "abalone"
BENCHMARK = True

# SUITE_ID = 335 # Regression on numerical and categorical features
# SUITE_ID = 336 # Regression on numerical features
# SUITE_ID = 334 # Classification on numerical and categorical features
SUITE_ID = 337 # Classification on numerical features

CLASSIFICATION = BENCHMARK and (SUITE_ID in [334, 337])

# Periodicity
MODEL = "tabbaseline"  # 'fnet', 'tabfnet', 'cnet', 'tabcnet', 'pnpnet', 'tabpnpnet', 'autopnpnet', 'tabautopnpnet', 'tabbaseline'
NUM_CHEBYSHEV_TERMS = 3
NUM_FOURIER_FEATURES = 15
HIDDEN_SIZE = 256

# Training
EPOCHS = 1000
LR = 0.01
BATCH_SIZE = 128
PATIENCE = 100

# Grid Search
CV_METRICS = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
MAX_ITERATIONS = 300

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

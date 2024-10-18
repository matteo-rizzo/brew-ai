# Common
DATASET = 'dataset.csv'
TARGET = "Tempo di riduzione diacetile"
BASE_LOG_DIR = "logs"
RANDOM_SEED = 0
NUM_FOLDS = 5
APPLY_PCA = False
TEST_SIZE = 0.2

# Deep Learning Models
MODEL_DEEP = "tabnet"  # 'tabnet', 'fttransformer', 'tabtransformer'
EPOCHS = 1000
LR = 0.001

# Grid Search
CV_METRICS = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
MAX_ITERATIONS = 300

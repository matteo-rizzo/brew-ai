import numpy as np
import torch
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf

from src.classes.data.DataPreprocessor import DataPreprocessor
from src.classes.periodicity.CrossValidator import CrossValidator
from src.functions.utils import load_data
from src.settings import APPLY_PCA, MODEL_PERIODICITY, RANDOM_SEED

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def detect_periodicity_acf(series, lag_limit=50):
    autocorr = acf(series, nlags=lag_limit, fft=True)
    peaks, _ = find_peaks(autocorr[1:])  # Exclude lag 0
    return len(peaks) > 0


def main():
    x, y = load_data()
    idx_periodic = []
    idx_non_periodic = []

    x_num = x[x.select_dtypes(include=['float64', 'int64']).columns.tolist()]
    for column in x_num.columns[:-1]:
        series = x_num[column].values
        if detect_periodicity_acf(series):
            idx_periodic.append(x_num.columns.get_loc(column))
        else:
            idx_non_periodic.append(x_num.columns.get_loc(column))

    idx_num = [x.columns.get_loc(col) for col in x.select_dtypes(include=['float64', 'int64']).columns.tolist()]
    idx_cat = [x.columns.get_loc(col) for col in x.select_dtypes(include=['object', 'category']).columns.tolist()]

    data_preprocessor = DataPreprocessor(x, y, apply_pca=APPLY_PCA)
    preprocessor = data_preprocessor.preprocess()
    x = preprocessor.fit_transform(x)

    # Create an instance of CrossValidator
    cross_validator = CrossValidator(
        model_name=MODEL_PERIODICITY,
        x=x,
        y=y,
        idx_num=idx_num,
        idx_cat=idx_cat,
        idx_periodic=idx_periodic,
        idx_non_periodic=idx_non_periodic,
        num_folds=5,
        batch_size=32,
        num_epochs=100,
        learning_rate=1e-3
    )

    # Run cross-validation
    results = cross_validator.run()


if __name__ == "__main__":
    main()

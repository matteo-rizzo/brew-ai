import numpy as np
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf


class PeriodicityDetector:

    @staticmethod
    def detect_periodicity_acf(series, lag_limit=50, peak_height=0.1, peak_prominence=0.1, min_distance=5):
        """
        Detects periodicity in a time series using the autocorrelation function (ACF).

        :param series: pd.Series or np.array, The time series data.
        :param lag_limit: int, Maximum number of lags to consider in ACF.
        :param peak_height: float, Minimum height for peaks in ACF to be considered.
        :param peak_prominence: float, Minimum prominence of peaks in ACF.
        :param min_distance: int, Minimum distance between peaks to avoid closely spaced false positives.
        :return: bool, True if periodicity is detected, False otherwise.
        """
        # Compute autocorrelation function
        autocorr = acf(series, nlags=lag_limit, fft=True)

        # Detect peaks in the autocorrelation, excluding lag 0
        peaks, _ = find_peaks(
            autocorr[1:], height=peak_height, prominence=peak_prominence, distance=min_distance
        )

        # Check periodicity by examining the distances between detected peaks
        if len(peaks) > 1:
            peak_distances = np.diff(peaks)
            # Check if peak distances are relatively uniform (standard deviation threshold)
            if np.std(peak_distances) < 2:  # Adjust threshold for tolerance
                return True

        return False

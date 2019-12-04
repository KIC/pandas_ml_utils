from pandas_ml_utils.constants import SIMULATED_VECTOR

import pandas as pd
import numpy as np


def _simulate_smoothing(features, lag_smoothing):
    simulated_frame = pd.DataFrame({f: SIMULATED_VECTOR for f in features})
    smoothing_length = 0

    for k, v in (lag_smoothing or {}).items():
        simulated_result = v(simulated_frame)
        nan_count = np.isnan(simulated_result.values if isinstance(simulated_result, (pd.Series, pd.DataFrame)) else simulated_result).sum()
        gap_len = len(simulated_frame) - len(simulated_result)
        smoothing_length = max(smoothing_length, gap_len + nan_count)

    return smoothing_length + 1


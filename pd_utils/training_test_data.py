import pandas as pd
import numpy as np
from typing import List, Tuple, Callable, Iterable, Dict


class Model(object):
    pass


class FeaturesAndLabels(object):

    def __init__(self,
                 features: List[str],
                 labels: List[str],
                 feature_lags: Iterable[int] = None,
                 lag_smoothing: Dict[int, Callable[[pd.Series], pd.Series]] = None,
                 probability_cutoff: float = 0.5):
        self.features = features
        self.labels = labels
        self.feature_lags = feature_lags
        self.lag_smoothing = lag_smoothing
        self.probability_cutoff = probability_cutoff


class Classification(object):
    pass



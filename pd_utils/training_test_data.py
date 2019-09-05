import pandas as pd
import numpy as np
from typing import List, Tuple, Callable, Iterable, Dict


class Model(object):
    pass


class FeaturesAndLabels(object):

    def __init__(self,
                 features: List[str],
                 labels: List[str],
                 target_columns: List[str] = None,
                 feature_lags: Iterable[int] = None,
                 lag_smoothing: Dict[int, Callable[[pd.Series], pd.Series]] = None,
                 probability_cutoff: float = 0.5):
        self.features = features
        self.labels = labels
        self.target_columns = target_columns
        self.feature_lags = feature_lags
        self.lag_smoothing = lag_smoothing
        self.probability_cutoff = probability_cutoff
        self.expanded_feature_length = len(features) * sum(1 for _ in feature_lags) if feature_lags is not None else len(features)

    def len_features(self):
        return len(self.features), self.expanded_feature_length

    def len_labels(self):
        return len(self.labels)

    def __repr__(self):
        return f'FeaturesAndLabels({self.features},{self.labels},{self.target_columns}{self.feature_lags},' \
               f'{self.lag_smoothing},{self.probability_cutoff}) #{len(self.features)}' \
               f' features expand to {self.expanded_feature_length}'

    def __str__(self):
        return self.__repr__()


class ClassificationSummary(object):

    def __init__(self,
                 y_true: np.ndarray,
                 y_prediction: np.ndarray,
                 index: np.ndarray,
                 probability_cutoff: float = 0.5):
        self.y_true = y_true
        self.y_prediction = y_prediction
        self.index = index
        self.probability_cutoff = probability_cutoff
        self.confusion_matrix = self._confusion_matrix_indices()

    def _confusion_matrix_indices(self):
        index = self.index
        truth = self.y_true
        pred = self.y_prediction
        co = self.probability_cutoff

        confusion = np.array([[index[(truth == True) & (pred > co)], index[(truth == False) & (pred > co)]],
                              [index[(truth == True) & (pred <= co)], index[(truth == False) & (pred <= co)]]])

        return confusion

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'\n{len(self.confusion_matrix[0,0])}\t{len(self.confusion_matrix[0,1])}' \
               f'\n{len(self.confusion_matrix[1,0])}\t{len(self.confusion_matrix[1,1])}'

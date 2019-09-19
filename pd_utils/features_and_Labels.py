import inspect
import logging
from typing import List, Callable, Iterable, Dict

import pandas as pd

log = logging.getLogger(__name__)


class FeaturesAndLabels(object):

    def __init__(self,
                 features: List[str],
                 labels: List[str],
                 target_columns: List[str] = None,
                 loss_column: str = None,
                 feature_lags: Iterable[int] = None,
                 lag_smoothing: Dict[int, Callable[[pd.Series], pd.Series]] = None,
                 probability_cutoff: float = 0.5):
        self.features = features
        self.labels = labels
        self.target_columns = target_columns
        self.loss_column = loss_column
        self.feature_lags = feature_lags
        self.lag_smoothing = lag_smoothing
        self.probability_cutoff = probability_cutoff
        self.len_feature_lags = sum(1 for _ in feature_lags) if feature_lags is not None else 1
        self.expanded_feature_length = len(features) * self.len_feature_lags if feature_lags is not None else len(features)
        log.info(f'number of features, lags and total: {self.len_features()}')

    def len_features(self):
        return len(self.features), self.len_feature_lags, self.expanded_feature_length

    def len_labels(self):
        return len(self.labels)

    def __repr__(self):
        return f'FeaturesAndLabels({self.features},{self.labels},{self.target_columns},{self.loss_column},' \
               f'{self.feature_lags},{self.lag_smoothing},{self.probability_cutoff}) #{len(self.features)} ' \
               f'features expand to {self.expanded_feature_length}'

    def __hash__(self):
        return hash(self.__id__())

    def __eq__(self, other):
        return self.__id__() == other.__id__()

    def __id__(self):
        import dill  # only import if really needed
        smoothers = {feature: inspect.getsource(smoother) for feature, smoother in self.lag_smoothing.items()}
        return f'{self.features},{self.labels},{dill.dumps(self.feature_lags)},{smoothers}'

    def __str__(self):
        return self.__repr__()


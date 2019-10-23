import inspect
import logging
from typing import List, Callable, Iterable, Dict, Type, Tuple

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


class FeaturesAndLabels(object):

    def __init__(self,
                 features: List[str],
                 labels: List[str],
                 label_type:Type = int,
                 target_columns: List[str] = None,
                 loss_column: str = None,
                 feature_lags: Iterable[int] = None,
                 feature_rescaling: Dict[Tuple[str], Tuple[int]] = None,
                 lag_smoothing: Dict[int, Callable[[pd.Series], pd.Series]] = None,
                 **kwargs):
        self.features = features
        self.labels = labels
        self.label_type = label_type
        self.target_columns = target_columns
        self.loss_column = loss_column
        self.feature_lags = feature_lags
        self.feature_rescaling = feature_rescaling
        self.lag_smoothing = lag_smoothing
        self.len_feature_lags = sum(1 for _ in feature_lags) if feature_lags is not None else 1
        self.expanded_feature_length = len(features) * self.len_feature_lags if feature_lags is not None else len(features)
        self.kwargs = kwargs
        log.info(f'number of features, lags and total: {self.len_features()}')

    def shape(self):
        return self.get_feature_names().shape, (self.len_labels(), )

    def len_features(self):
        return len(self.features), self.len_feature_lags, self.expanded_feature_length

    def len_labels(self):
        return len(self.labels)

    def get_feature_names(self):
        if self.feature_lags is not None:
            return np.array([[f'{feat}_{lag}'
                              for feat in self.features]
                             for lag in self.feature_lags], ndmin=2)
        else:
            return np.array(self.features)

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == 2:
            return self.kwargs[item[0]] if item[0] in self.kwargs else item[1]
        else:
            return self.kwargs[item] if item in self.kwargs else None

    def __repr__(self):
        return f'FeaturesAndLabels({self.features},{self.labels},{self.target_columns},{self.loss_column},' \
               f'{self.feature_lags},{self.feature_rescaling}{self.lag_smoothing},{self.probability_cutoff}) ' \
               f'#{len(self.features)} ' \
               f'features expand to {self.expanded_feature_length}'

    def __hash__(self):
        return hash(self.__id__())

    def __eq__(self, other):
        return self.__id__() == other.__id__()

    def __id__(self):
        import dill  # only import if really needed
        smoothers = ""

        if self.lag_smoothing is not None:
            smoothers = {feature: inspect.getsource(smoother) for feature, smoother in self.lag_smoothing.items()}

        return f'{self.features},{self.labels},{self.label_type},{dill.dumps(self.feature_lags)},{self.feature_rescaling},{smoothers}'

    def __str__(self):
        return self.__repr__()


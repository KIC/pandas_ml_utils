import inspect
import logging
from typing import List, Callable, Iterable, Dict, Type, Tuple, Union
from numbers import Number
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


class FeaturesAndLabels(object):

    def __init__(self,
                 features: List[str],
                 labels: List[str],
                 label_type:Type = int,
                 targets: Union[List[str], Tuple[str, str], Dict[str, str], Dict[str, Tuple[str, List[str]]], Dict[str, Tuple[str, str]]] = None,
                 feature_lags: Iterable[int] = None,
                 feature_rescaling: Dict[Tuple[str], Tuple[int]] = None,
                 lag_smoothing: Dict[int, Callable[[pd.Series], pd.Series]] = None,
                 **kwargs):
        self.features = features
        self.labels = labels
        self.label_type = label_type
        self.targets = targets
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

    def get_goals(self) -> Dict[str, Tuple[str, List[str]]]:
        # if we can return a dictionary of target -> (loss, labels) where loss will be a column or constant 1
        if isinstance(self.targets, Number) or isinstance(self.targets, str):
            # we have a target value but no loss
            return {self.targets: (None, self.labels)}
        elif self.targets is None or len(self.targets) <= 0:
            # we have got no target no loss an no labels
            return {None: (None, self.labels)}
        elif isinstance(self.targets, tuple):
            # we have a pair of target value and loss
            return {self.targets[0]: (self.targets[1], self.labels)}
        elif isinstance(self.targets, dict):
            sample_value = next(iter(self.targets.values()))
            if isinstance(sample_value, tuple) and len(sample_value) == 2 and isinstance(sample_value[1], list):
                # we have a dict of target values and tuple losses, labels
                return self.targets
            elif isinstance(sample_value, tuple) and len(sample_value) == 2 and isinstance(sample_value[1], str):
                # we have a dict of target values and tuple losses, labels
                return {k: (v[0], [v[1]]) for k, v in self.targets.items()}
            elif isinstance(sample_value, str) or isinstance(sample_value, Number):
                # we have a dict of target values and losses
                return {k: (v, self.labels) for k, v in self.targets.items()}
            else:
                raise ValueError("you need to provide a loss or a tuple[loss, list[str]")
        else:
            raise ValueError("you need to provide a traget column name oder a dictionary with target column name as key")

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == 2:
            return self.kwargs[item[0]] if item[0] in self.kwargs else item[1]
        else:
            return self.kwargs[item] if item in self.kwargs else None

    def __repr__(self):
        return f'FeaturesAndLabels({self.features},{self.labels},{self.targets},' \
               f'{self.feature_lags},{self.feature_rescaling}{self.lag_smoothing}) ' \
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

        return f'{self.features},{self.labels},{self.label_type},{self.targets},{dill.dumps(self.feature_lags)},{self.feature_rescaling},{smoothers}'

    def __str__(self):
        return self.__repr__()


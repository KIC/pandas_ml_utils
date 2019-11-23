import inspect
import logging
from typing import List, Callable, Iterable, Dict, Type, Tuple, Union
from numbers import Number
import pandas as pd
import numpy as np

_log = logging.getLogger(__name__)
_SIMULATED_VECTOR = np.ones(10000)


class FeaturesAndLabels(object):
    """
    *FeaturesAndLabels* is the main object used to hold the context of your problem. Here you define which columns
    of your `DataFrame` is a feature, a label or a target. This class also provides some functionality to generate
    autoregressive features. By default lagging features results in an RNN shaped 3D array (in the format of keras
    RNN layers input format).
    """

    def __init__(self,
                 features: List[str],
                 labels: List[str],
                 label_type:Type = int,
                 targets: Union[List[str], Tuple[str, str], Dict[str, str], Dict[str, Tuple[str, List[str]]], Dict[str, Tuple[str, str]]] = None,
                 feature_lags: Iterable[int] = None,
                 feature_rescaling: Dict[Tuple[str], Tuple[int]] = None,
                 lag_smoothing: Dict[int, Callable[[pd.Series], pd.Series]] = None,
                 **kwargs):
        """
        :param features: a list of column names which are used as features for your model
        :param labels: as list of column names which are uses as labels for your model
        :param label_type: whether to treat a label as int, float, bool
        :param targets: targets are composed of a *target*, *loss*, *subset of labels* where any part can be None.
                        Technically at least one target is needed and defaults to ("target", -1, self.labels) but
                        multiple targets are possible. The concept around targets is like follows. Let's say you want
                        to classify whether a printer is jamming the next page or not. Halting and servicing the printer
                        costs 5'000 while a jam costs 15'000. Your target will be 0 or empty but your loss will be
                        -5000 for all your type II errors and -15'000 for all your type I errors in case of miss-
                        classification. Another example would be if you want to classify whether a stock price is
                        above (buy) the current price or not (do nothing). Your target is the today's price and your
                        loss is tomorrows price minus today's price.
                        Note: Not all :class:`.Model` support multiple targets
        :param feature_lags: an iterable of integers specifying the lags of an AR model i.e. [1] for AR(1)
                             if the un-lagged feature is needed as well provide also lag of 0 like range(1)
        :param feature_rescaling: this allows to rescale features.
                                  in a dict we can define a tuple of column names and a target range
        :param lag_smoothing: very long lags in an AR model can be a bit fuzzy, it is possible to smooth lags i.e. by
                              using moving averages. the key is the lag length at which a smoothing function starts to
                              be applied
        :param kwargs: maybe you want to pass some extra parameters to a model
        """
        self.features = features
        self.labels = labels
        self.label_type = label_type
        self.targets = targets
        self.feature_lags = [lag for lag in feature_lags] if feature_lags is not None else None
        self.feature_rescaling = feature_rescaling
        self.lag_smoothing = lag_smoothing
        self.len_feature_lags = sum(1 for _ in self.feature_lags) if self.feature_lags is not None else 1
        self.expanded_feature_length = len(features) * self.len_feature_lags if feature_lags is not None else len(features)
        self.min_required_samples = (max(feature_lags) + self.__simulate_smoothing()) if self.feature_lags is not None else 1
        self.kwargs = kwargs
        _log.info(f'number of features, lags and total: {self.len_features()}')

    @property
    def shape(self) -> Tuple[Tuple[int], Tuple[int]]:
        """
        Returns the shape of features and labels how they get passed to the :class:`.Model`. If laging is used, then
        the features shape is in Keras RNN form.

        :return: a tuple of (features.shape, labels.shape)
        """

        return self.get_feature_names().shape, (self.len_labels(), )

    def len_features(self) -> Tuple[int]:
        """
        Returns the length of the defined features, the number of lags used and the total number of all features * lags

        :return: tuple of (#features, #lags, #features * #lags)
        """

        return len(self.features), self.len_feature_lags, self.expanded_feature_length

    def len_labels(self) -> int:
        """
        Returns the number of labels

        :return:  number of labels
        """
        return len(self.labels)

    def get_feature_names(self) -> np.ndarray:
        """
        Returns all features names eventually post-fixed with the length of the lag

        :return: numpy array of strings in the shape of the features
        """
        if self.feature_lags is not None:
            return np.array([[f'{feat}_{lag}'
                              for feat in self.features]
                             for lag in self.feature_lags], ndmin=2)
        else:
            return np.array(self.features)

    def get_goals(self) -> Dict[str, Tuple[str, List[str]]]:
        """
        A Goal is the combination of the target and its loss and labels

        :return: all goals for all given targets
        """
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

    def __simulate_smoothing(self):
        simulated_frame = pd.DataFrame({f: _SIMULATED_VECTOR for f in self.features})
        smoothing_length = 0

        for k, v in (self.lag_smoothing or {}).items():
            simulated_result = v(simulated_frame)
            nan_count = np.isnan(simulated_result.values if isinstance(simulated_result, (pd.Series, pd.DataFrame)) else simulated_result).sum()
            gap_len = len(simulated_frame) - len(simulated_result)
            smoothing_length = max(smoothing_length, gap_len + nan_count)

        return smoothing_length + 1

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


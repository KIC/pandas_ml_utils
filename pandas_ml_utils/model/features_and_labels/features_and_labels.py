import inspect
import logging
from typing import List, Callable, Iterable, Dict, Type, Tuple, Union, Any
from numbers import Number
from pandas_ml_utils.model.features_and_labels.target_encoder import TargetLabelEncoder
from pandas_ml_utils.model.features_and_labels.sample_size_estimator import _simulate_smoothing
import pandas as pd
import numpy as np

_log = logging.getLogger(__name__)


# This class should be able to be pickled and unpickled without risk of change between versions
# This means business logic need to be kept outside of this class!
class FeaturesAndLabels(object):
    """
    *FeaturesAndLabels* is the main object used to hold the context of your problem. Here you define which columns
    of your `DataFrame` is a feature, a label or a target. This class also provides some functionality to generate
    autoregressive features. By default lagging features results in an RNN shaped 3D array (in the format of keras
    RNN layers input format).
    """

    def __init__(self,
                 features: List[str],
                 labels: Union[List[str], TargetLabelEncoder, Dict[str, Union[List[str], TargetLabelEncoder]]],
                 label_type:Type = int,
                 loss: Callable[[str, pd.DataFrame], Union[pd.Series, pd.DataFrame]] = None,
                 targets: Callable[[str, pd.DataFrame], Union[pd.Series, pd.DataFrame]] = None,
                 feature_lags: Iterable[int] = None,
                 feature_rescaling: Dict[Tuple[str, ...], Tuple[int, ...]] = None, # fiXme lets provide a rescaler ..
                 lag_smoothing: Dict[int, Callable[[pd.Series], pd.Series]] = None,
                 **kwargs):
        """
        :param features: a list of column names which are used as features for your model
        :param labels: as list of column names which are uses as labels for your model. you can specify one ore more
                       named targets for a set of labels by providing a dict. This is useful if you want to train a
                       :class:`.MultiModel` or if you want to provide extra information about the label. i.e. you
                       want to classify whether a stock price is bleow or above average and you want to provide what
                       the average was.
        :param label_type: whether to treat a label as int, float, bool
        :param loss: Let's say you want to classify whether a printer is jamming the next page or not. Halting and
                     servicing the printer costs 5'000 while a jam costs 15'000. Your target will be 0 or empty but
                     your loss will be -5000 for all your type II errors and -15'000 for all your type I errors in
                     case of miss-classification. Another example would be if you want to classify whether a stock
                     price is above (buy) the current price or not (do nothing). Your target is the today's price
                     and your loss is tomorrows price minus today's price.
        :param targets: expects a callable which receives a target (or None) and the source data frame and should
                        return a series or data frame. In case of multiple targets the series names need to be unique!
        :param feature_lags: an iterable of integers specifying the lags of an AR model i.e. [1] for AR(1)
                             if the un-lagged feature is needed as well provide also lag of 0 like range(1)
        :param feature_rescaling: this allows to rescale features.
                                  in a dict we can define a tuple of column names and a target range
        :param lag_smoothing: very long lags in an AR model can be a bit fuzzy, it is possible to smooth lags i.e. by
                              using moving averages. the key is the lag length at which a smoothing function starts to
                              be applied
        :param kwargs: maybe you want to pass some extra parameters to a model
        """
        self._features = features
        self._labels = labels
        self._targets = targets
        self._loss = loss
        self.label_type = label_type
        self.feature_lags = [lag for lag in feature_lags] if feature_lags is not None else None
        self.feature_rescaling = feature_rescaling
        self.lag_smoothing = lag_smoothing
        self.len_feature_lags = sum(1 for _ in self.feature_lags) if self.feature_lags is not None else 1
        self.expanded_feature_length = len(features) * self.len_feature_lags if feature_lags is not None else len(features)
        self.min_required_samples = (max(feature_lags) + _simulate_smoothing(features, lag_smoothing)) if self.feature_lags is not None else 1
        self.kwargs = kwargs
        _log.info(f'number of features, lags and total: {self.len_features()}')

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def targets(self):
        return self._targets

    @property
    def loss(self):
        return self._loss

    @property
    def shape(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Returns the shape of features and labels how they get passed to the :class:`.Model`. If laging is used, then
        the features shape is in Keras RNN form.

        :return: a tuple of (features.shape, labels.shape)
        """

        return self.get_feature_names().shape, (self.len_labels(), )

    def len_features(self) -> Tuple[int, ...]:
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

    #@deprecation.deprecated()
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



import logging
from functools import lru_cache
from time import perf_counter as pc
from typing import Tuple, Dict, Union, List

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict

from pandas_ml_utils.constants import *
from pandas_ml_utils.model.features_and_labels.features_and_labels import FeaturesAndLabels
from pandas_ml_utils.model.features_and_labels.target_encoder import TargetLabelEncoder, \
    MultipleTargetEncodingWrapper, IdentityEncoder
from pandas_ml_utils.utils.classes import ReScaler
from pandas_ml_utils.utils.functions import log_with_time, call_callable_dynamic_args

_log = logging.getLogger(__name__)


class FeatureTargetLabelExtractor(object):

    def __init__(self, df: pd.DataFrame, features_and_labels: FeaturesAndLabels, **kwargs):
        # prepare fields
        labels = features_and_labels.labels
        encoder = lambda frame: frame
        label_columns = None

        # eventually transform callable labels to its expected structure
        if callable(labels):
            joined_kwargs = {**features_and_labels.kwargs, **kwargs}
            labels = call_callable_dynamic_args(labels, df, **joined_kwargs)

        # unfold labels, currently supported types are:
        #  Union[List[str], TargetLabelEncoder, Dict[str, Union[List[str], TargetLabelEncoder]]]
        if isinstance(labels, list):
            label_columns = labels
        elif isinstance(labels, TargetLabelEncoder):
            encoder = labels.encode
            label_columns = labels.labels_source_columns
        elif isinstance(labels, Dict):
            # this is our multi model case, here we add an extra dimension to the labels array
            label_columns = [l for ls in labels.values()
                             for l in (ls if isinstance(ls, list) else ls.labels_source_columns)]

            # we need a special encoder which is wrapping all encoder for each target
            encoder = MultipleTargetEncodingWrapper({
                t: l if isinstance(l, TargetLabelEncoder) else IdentityEncoder(l) for t, l in labels.items()
            }).encode

        # assign all fields
        self._features_and_labels = features_and_labels # depricated copy all fields here
        self._features = features_and_labels.features
        self._labels_columns = label_columns
        self._labels = labels
        self._label_type = features_and_labels.label_type
        self._targets = features_and_labels.targets
        self._gross_loss = features_and_labels.gross_loss
        self._encoder = encoder

        # pre assign this variable
        # but notice that it get overwritten by an engineered data frame later on
        self.df = df

        # this function uses clojures
        def call_dynamic(func, *args):
            joined_kwargs = {**self.__dict__, **features_and_labels.kwargs, **kwargs}
            return call_callable_dynamic_args(func, *args, **joined_kwargs)

        self.df = call_dynamic(features_and_labels.pre_processor, df)
        self.__call_dynamic = call_dynamic

    def prediction_to_frame(self,
                            prediction: np.ndarray,
                            index: pd.Index = None,
                            inclusive_labels: bool = False,
                            inclusive_source: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        # sanity check
        if not isinstance(prediction, np.ndarray):
            raise ValueError(f"got unexpected prediction: {type(prediction)}\n{prediction}")

        # assign index
        index = self.df.index if index is None else index

        # eventually fix the shape of the prediction
        if len(prediction.shape) == 1:
            prediction = prediction.reshape(len(prediction), 1)

        # prediction_columns
        df = pd.DataFrame(prediction, index=index, columns=pd.MultiIndex.from_tuples(self.label_names(PREDICTION_COLUMN_NAME)))

        # add labels if requested
        if inclusive_labels:
            dfl = self.labels_df
            dfl.columns = pd.MultiIndex.from_tuples(self.label_names(LABEL_COLUMN_NAME))
            df = df.join(dfl, how='inner')

            # add loss if provided
            loss_df = self.gross_loss_df
            df = df.join(loss_df.loc[df.index], how='inner') if loss_df is not None else df

        # add target if provided
        target_df = self.target_df
        df = df.join(target_df.loc[df.index], how='inner') if target_df is not None else df

        # also add source if requested
        if inclusive_source:
            df = df.join(self.source_df, how='inner')

        # finally we can return our nice and shiny df
        return df

    @property
    def features(self) -> Tuple[pd.DataFrame, np.ndarray]:
        df = self.features_df
        x = self._fix_shape(df)

        _log.info(f"features shape: {x.shape}")
        return df, x

    @property
    def min_required_samples(self):
        return len(self.df) - len(self.features_df) + 1

    @property
    def features_labels(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        # engineer features and labels
        df_features = self.features_df
        df_labels = self.labels_df
        df = self.features_df.join(df_labels, how='inner').dropna()

        # select only joining index values
        df_features = df_features.loc[df.index]
        df_labels = df_labels.loc[df.index]

        # features eventually are in RNN shape which is [row, time_step, feature]
        x = self._fix_shape(df_features)

        # labels are straight forward but eventually need to be type corrected
        y = df_labels.values.astype(self._label_type)
        _log.info(f"  features shape: {x.shape}, labels shape: {y.shape}")

        # sanity check
        if not len(x) == len(y) == len(df):
            raise ValueError(f"unbalanced length of features and labels {len(x), len(y), len(df)}")

        return df, x, y

    @property
    @lru_cache(maxsize=1)
    def features_df(self) -> pd.DataFrame:
        start_pc = log_with_time(lambda: _log.debug(" make features ..."))
        feature_lags = self._features_and_labels.feature_lags
        features = self._features
        lag_smoothing = self._features_and_labels.lag_smoothing
        feature_rescaling = self._features_and_labels.feature_rescaling

        # drop nan's and copy frame
        df = self.df[features].dropna().copy()

        # generate feature matrix
        if feature_lags is None:
            dff = df
        else:
            dff = pd.DataFrame({}, index=df.index)
            # return RNN shaped 3D arrays
            for feature in features:
                feature_series = df[feature]
                smoothers = None

                # smooth out feature if requested
                if lag_smoothing is not None:
                    smoothers = SortedDict({lag: smoother(feature_series.to_frame())
                                            for lag, smoother in lag_smoothing.items()})

                for lag in feature_lags:
                    # if smoothed values are applicable use smoothed values
                    if smoothers is not None and len(smoothers) > 0 and smoothers.peekitem(0)[0] <= lag:
                        feature_series = smoothers.popitem(0)[1]

                    # assign the lagged (eventually smoothed) feature to the features frame
                    dff[(feature, lag)] = feature_series.shift(lag)

            # fix tuple column index to actually be a multi index
            dff.columns = pd.MultiIndex.from_tuples(dff.columns)

            # drop all rows which got nan now
            dff = dff.dropna()

        # do rescaling
        if feature_rescaling is not None:
            for rescale_features, target_range in feature_rescaling.items():
                # tuple need to be converted to list!
                rescale_features = [f for f in rescale_features]

                # multi index has problems in the direct assignent so we need to copy back column by column
                tmp = dff[rescale_features].apply(lambda row: ReScaler((row.min(), row.max()), target_range)(row),
                                                  raw=True, result_type='broadcast')
                for col in tmp.columns:
                    dff[col] = tmp[col]

        _log.info(f" make features ... done in {pc() - start_pc: .2f} sec!")
        return dff

    @property
    def feature_names(self) -> np.ndarray:
        return np.array(self._features)

    def label_names(self, level_above=None) -> List[Union[Tuple[str, ...],str]]:
        labels = self._labels.encoded_labels_columns \
            if isinstance(self._labels, TargetLabelEncoder) else self._labels

        if isinstance(labels, dict):
            label_columns = []
            for target, target_labels in labels.items():
                for label in (target_labels.encoded_labels_columns if isinstance(target_labels, TargetLabelEncoder) else target_labels):
                    label_columns.append((target, label) if level_above is None else (target, level_above, label))

            return label_columns
        else:
            return labels if level_above is None else [(level_above, col) for col in labels]

    @property
    def labels_df(self) -> pd.DataFrame:
        # here we can do all sorts of tricks and encodings ...
        df = self._encoder(self.df[self._labels_columns]).dropna().copy()
        return df

    @property
    def source_df(self):
        df = self.df.copy()
        df.columns = pd.MultiIndex.from_product([[SOURCE_COLUMN_NAME], df.columns])
        return df

    @property
    def gross_loss_df(self):
        df = None

        if self._gross_loss is not None:
            labels = self._labels
            for target in (labels.keys() if isinstance(labels, dict) else [None]):
                dfl = self.__call_dynamic(self._gross_loss, self.df, target)
                if isinstance(dfl, pd.Series):
                    if dfl.name is None:
                        dfl.name = target or GROSS_LOSS_COLUMN_NAME
                    dfl = dfl.to_frame()

                dfl.columns = [(GROSS_LOSS_COLUMN_NAME, col) if target is None else (target, GROSS_LOSS_COLUMN_NAME, col)
                               for col in dfl.columns]

                df = dfl if df is None else df.join(dfl)

            # multi level index
            df.columns = pd.MultiIndex.from_tuples(df.columns)

        return df

    @property
    def target_df(self):
        df = None

        if self._targets is not None:
            labels = self._labels
            for i, target in enumerate(labels.keys() if isinstance(labels, dict) else [None]):
                dft = self.__call_dynamic(self._targets, self.df, target)

                if isinstance(dft, pd.Series):
                    if dft.name is None:
                        dft.name = target or TARGET_COLUMN_NAME
                    dft = dft.to_frame()
                elif not isinstance(dft, (pd.Series, pd.DataFrame)):
                    dft = pd.DataFrame({target or TARGET_COLUMN_NAME: dft}, index=self.df.index)

                dft.columns = [(TARGET_COLUMN_NAME, col) if target is None else (target, TARGET_COLUMN_NAME, col)
                               for col in dft.columns]

                df = dft if df is None else df.join(dft)

            # multi level index
            df.columns = pd.MultiIndex.from_tuples(df.columns)

        return df

    def _fix_shape(self, df_features):
        # features eventually are in RNN shape which is [row, time_step, feature]
        feature_arr = df_features.values if self._features_and_labels.feature_lags is None else \
            np.array([df_features[cols].values for cols in self.feature_names], ndmin=3).swapaxes(0, 1).swapaxes(1, 2)

        if len(feature_arr) <= 0:
            _log.warning("empty feature array!")

        return feature_arr

    def __str__(self):
        return f'min required data = {self.min_required_samples}'

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
from pandas_ml_utils.model.fitting.splitting import train_test_split
from pandas_ml_utils.utils.classes import ReScaler
from pandas_ml_utils.utils.functions import log_with_time, call_callable_dynamic_args, unique_top_level_columns, \
    join_kwargs, integrate_nested_arrays

_log = logging.getLogger(__name__)


class FeatureTargetLabelExtractor(object):

    def __init__(self, df: pd.DataFrame, features_and_labels: FeaturesAndLabels, **kwargs):
        # prepare fields
        labels = features_and_labels.labels
        encoder = lambda frame, **kwargs: frame
        label_columns = None
        joined_kwargs = join_kwargs(features_and_labels.kwargs, kwargs)

        # eventually transform callable labels to its expected structure
        if callable(labels):
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
        self._joined_kwargs = joined_kwargs

        # pre assign this variable
        # but notice that it get overwritten by an engineered data frame later on
        self._df = df

        # this function uses clojures
        def call_dynamic(func, *args):
            joined_kwargs = join_kwargs(self.__dict__, self._joined_kwargs)
            return call_callable_dynamic_args(func, *args, **joined_kwargs)

        self._df = call_dynamic(features_and_labels.pre_processor, df)
        self.__call_dynamic = call_dynamic

    @property
    def df(self):
        return self._df

    @property
    def min_required_samples(self):
        return len(self._df) - len(self.features_df) + 1

    def prediction_to_frame(self,
                            prediction: np.ndarray,
                            index: pd.Index = None,
                            inclusive_labels: bool = False,
                            inclusive_source: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        # sanity check
        if not isinstance(prediction, np.ndarray):
            raise ValueError(f"got unexpected prediction: {type(prediction)}\n{prediction}")

        # assign index
        index = self._df.index if index is None else index

        # eventually fix the shape of the prediction
        if len(prediction.shape) == 1:
            prediction = prediction.reshape(len(prediction), 1)

        # prediction_columns
        columns = pd.MultiIndex.from_tuples(self.label_names(PREDICTION_COLUMN_NAME))
        multi_dimension_prediction = len(prediction.shape) > 1 and len(columns) < prediction.shape[1]
        if multi_dimension_prediction:
            if len(prediction.shape) < 3:
                df = pd.DataFrame({"a":[ r.tolist() for r in prediction]}, index=index)
            else:
                df = pd.DataFrame({col: [row.tolist() for row in prediction[:, col]] for col in range(prediction.shape[1])},index=index)

            df.columns = columns
        else:
             df = pd.DataFrame(prediction, index=index, columns=columns)

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

    def training_and_test_data(self,
                               test_size: float = 0.4,
                               youngest_size: float = None,
                               seed: int = 42) -> Tuple[Tuple[np.ndarray,...], Tuple[np.ndarray,...]]:
        features, labels, weights = self.features_labels_weights_df
        train_ix, test_ix = train_test_split(features.index, test_size, youngest_size, seed=seed)

        return (
            (train_ix,
             features.loc[train_ix].values,
             integrate_nested_arrays(labels.loc[train_ix].values),
             weights.loc[train_ix].values if weights is not None else None),
            (test_ix,
             features.loc[test_ix].values,
             integrate_nested_arrays(labels.loc[test_ix].values),
             weights.loc[test_ix].values if weights is not None else None)
        )

    @property
    def features_labels_weights_df(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # engineer features and labels
        df_features = self.features_df
        df_labels = self.labels_df
        index_intersect = df_features.index.intersection(df_labels.index)

        # select only joining index values
        df_features = df_features.loc[index_intersect]
        df_labels = df_labels.loc[index_intersect]
        # TODO add proper label weights
        df_weights = None #pd.DataFrame(np.ones(len(df_labels)), index=df_labels.index)

        # sanity check
        if not len(df_features) == len(df_labels):
            raise ValueError(f"unbalanced length of features and labels {len(df_features), len(df_labels)}")

        return df_features, df_labels, df_weights

    @property
    @lru_cache(maxsize=1)
    def features_df(self) -> pd.DataFrame:
        start_pc = log_with_time(lambda: _log.debug(" make features ..."))
        feature_lags = self._features_and_labels.feature_lags
        features = self._features
        lag_smoothing = self._features_and_labels.lag_smoothing
        feature_rescaling = self._features_and_labels.feature_rescaling

        # drop nan's and copy frame
        df = self._df[features].dropna().copy()

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

        # finally patch the "values" property for features data frame and return
        dff.__class__ = _RNNShapedValuesDataFrame
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
        # joined_kwargs(self._features_and_labels.kwargs, self.)
        df = self._encoder(self._df[self._labels_columns], **self._joined_kwargs).dropna().copy()
        return df if self._label_type is None else df.astype(self._label_type)

    @property
    def source_df(self):
        df = self._df.copy()
        df.columns = pd.MultiIndex.from_product([[SOURCE_COLUMN_NAME], df.columns])
        return df

    @property
    def gross_loss_df(self):
        df = None

        if self._gross_loss is not None:
            labels = self._labels
            for target in (labels.keys() if isinstance(labels, dict) else [None]):
                dfl = self.__call_dynamic(self._gross_loss, self._df, target)
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
                dft = self.__call_dynamic(self._targets, self._df, target)

                if isinstance(dft, pd.Series):
                    if dft.name is None:
                        dft.name = target or TARGET_COLUMN_NAME
                    dft = dft.to_frame()
                elif not isinstance(dft, (pd.Series, pd.DataFrame)):
                    dft = pd.DataFrame({target or TARGET_COLUMN_NAME: dft}, index=self._df.index)

                dft.columns = [(TARGET_COLUMN_NAME, col) if target is None else (target, TARGET_COLUMN_NAME, col)
                               for col in dft.columns]

                df = dft if df is None else df.join(dft)

            # multi level index
            df.columns = pd.MultiIndex.from_tuples(df.columns)

        return df

    def _fix_shape(self, df_features):
        # features eventually are in [feature, row, time_step]
        # but need to be in RNN shape which is [row, time_step, feature]
        feature_arr = df_features.values if self._features_and_labels.feature_lags is None else \
            np.array([df_features[cols].values for cols in self.feature_names], ndmin=3).swapaxes(0, 1).swapaxes(1, 2)

        if len(feature_arr) <= 0:
            _log.warning("empty feature array!")

        return feature_arr

    def __str__(self):
        return f'min required data = {self.min_required_samples}'


class _RNNShapedValuesDataFrame(pd.DataFrame):

    class Loc():
        def __init__(self, df):
            self.df = df

        def __getitem__(self, item):
            res = self.df.loc[item]
            res.__class__ = _RNNShapedValuesDataFrame
            return res

    @property
    def loc(self):
        return _RNNShapedValuesDataFrame.Loc(super(pd.DataFrame, self))

    @property
    def values(self):
        top_level_columns = unique_top_level_columns(self)

        # we need to do a sneaky trick here to get a proper "super" object as super() does not work as expected
        # so we simply rename with an empty dict
        df = self.rename({})

        # features eventually are in [feature, row, time_step]
        # but need to be in RNN shape which is [row, time_step, feature]
        feature_arr = df.values if top_level_columns is None else \
            np.array([df[feature].values for feature in top_level_columns],
                     ndmin=3).swapaxes(0, 1).swapaxes(1, 2)

        if len(feature_arr) <= 0:
            _log.warning("empty feature array!")

        return feature_arr

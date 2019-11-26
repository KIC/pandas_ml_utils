import logging
import re
from time import perf_counter as pc
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict

from pandas_ml_utils.constants import *
from pandas_ml_utils.model.features_and_Labels import FeaturesAndLabels
from pandas_ml_utils.model.features_and_labels_utils.target_encoder import TargetLabelEncoder
from pandas_ml_utils.utils.classes import ReScaler
from pandas_ml_utils.utils.functions import log_with_time

_log = logging.getLogger(__name__)


class FeatureTargetLabelExtractor(object):

    def __init__(self, df: pd.DataFrame, features_and_labels: FeaturesAndLabels):
        labels = features_and_labels.labels
        targets = {}

        # Union[List[str], Tuple[str, List[str]], TargetLabelEncoder, Dict[str, Union[List[str], TargetLabelEncoder]]]
        if isinstance(labels, list):
            targets[tuple(labels)] = labels
        elif isinstance(features_and_labels.labels, tuple):
            if len(labels[1]) > 1:
                targets[tuple([f"{l} #{i}" for i, l in enumerate(labels[0])])] = labels[1]
            else:
                targets[tuple([labels[0]])] = labels[1]
        elif isinstance(features_and_labels.labels, TargetLabelEncoder):
            # TODO targets (i.e. one hot names) and labels will come from TargetLabelEncoder
            pass
        elif isinstance(features_and_labels.labels, dict):
            # this is our multi model case, here we add an extra dimension to the labels array
            # TODO implement
            pass

        label_columns = [label for labels_lists in targets.values() for label in labels_lists]
        self.df = df
        self._features_and_labels = features_and_labels
        self._labels = label_columns
        self._targets = targets

    def prediction_to_frame(self, prediction: np.ndarray, index: pd.Index = None, inclusive_labels: bool = False) -> Dict[str, pd.DataFrame]:
        index = self.df.index if index is None else index
        frames = {}

        for target, df in self.targets.items():
            # create a data frame from the prediction
            df = pd.DataFrame({l: prediction[:, i] for i, l in enumerate(self._labels)}, index=index) if len(self._labels) > 1 \
                 else pd.DataFrame({self._labels[0]: prediction[:, 0] if len(prediction.shape) > 1 else prediction}, index=index)

            df.columns = pd.MultiIndex.from_arrays([[PREDICTION_COLUMN_NAME] * len(df.columns), df.columns])

            # add labels if requested
            if inclusive_labels:
                dfl = self.labels_df
                dfl.columns = pd.MultiIndex.from_arrays([[LABEL_COLUMN_NAME] * len(dfl.columns), dfl.columns])
                df = df.join(dfl, how='inner')

            # add loss if provided
            if self._features_and_labels.loss is not None:
                dfl = self._features_and_labels.loss(target, self.df.loc[df.index])
                if isinstance(dfl, pd.Series):
                    if dfl.name is None:
                        dfl.name = LOSS_COLUMN_NAME
                    dfl = dfl.to_frame()

                dfl.columns = pd.MultiIndex.from_arrays([[LOSS_COLUMN_NAME] * len(dfl.columns), dfl.columns])
                df = df.join(dfl, how='inner')

            frames[target] = df

        return frames

    @property
    def targets(self) -> Dict[str, pd.DataFrame]:
        # here we can do all the magic needed for the targets
        return {target: self.df[labels] for target, labels in self._targets.items()}

    @property
    def features(self) -> Tuple[pd.DataFrame, np.ndarray]:
        df = self.features_df
        x = df[self.feature_names].values if self._features_and_labels.feature_lags is None else \
            np.array([df[cols].values for cols in self.feature_names], ndmin=3).swapaxes(0, 1)

        _log.info(f"features shape: {x.shape}")
        return df, x

    @property
    def features_labels(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        df_labels = self.labels_df
        df = self.features_df.join(df_labels).dropna()

        # features eventually are in RNN shape which is [row, time_step, feature]
        x = df[self.feature_names].values if self._features_and_labels.feature_lags is None else \
            np.array([df[cols].values for cols in self.feature_names], ndmin=3).swapaxes(0, 1)

        # labels are straight forward but eventually need to be type corrected
        # TODO if len(targets > 1) then we need an extra dimension for y array
        y = df[df_labels.columns].values.astype(self._features_and_labels.label_type)

        _log.info(f"  features shape: {x.shape}, labels shape: {y.shape}")
        return df, x, y

    @property
    def features_df(self) -> pd.DataFrame:
        start_pc = log_with_time(lambda: _log.debug(" make features ..."))
        feature_lags = self._features_and_labels.feature_lags
        features = self._features_and_labels.features
        lag_smoothing = self._features_and_labels.lag_smoothing
        feature_rescaling = self._features_and_labels.feature_rescaling

        # drop nan's and copy frame
        df = self.df[features].dropna().copy()

        # generate feature matrix
        if feature_lags is not None:
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
                    df[f'{feature}_{lag}'] = feature_series.shift(lag)

            # drop all rows which got nan now
            df = df.dropna()

        # do rescaling
        if feature_rescaling is not None:
            for rescale_features, target_range in feature_rescaling.items():
                columns = [[col for col in df.columns for feature in rescale_features if re.match(rf"^{feature}(_\d+)?$", col)]]
                df[columns] = df[columns].apply(lambda row: ReScaler((row.min(), row.max()), target_range)(row),
                                                raw=True, result_type='broadcast')

        _log.info(f" make features ... done in {pc() - start_pc: .2f} sec!")
        return df

    @property
    def feature_names(self) -> np.ndarray:
        if self._features_and_labels.feature_lags is not None:
            return np.array([[f'{feat}_{lag}'
                              for feat in self._features_and_labels.features]
                             for lag in self._features_and_labels.feature_lags], ndmin=2)
        else:
            return np.array(self._features_and_labels.features)

    @property
    def labels_df(self) -> pd.DataFrame:
        # LATER here we can do all sorts of tricks and encodings ...
        df = self.df[self._labels].dropna().copy()
        return df

    def __str__(self):
        return f'min required data = {self._features_and_labels.min_required_samples}'

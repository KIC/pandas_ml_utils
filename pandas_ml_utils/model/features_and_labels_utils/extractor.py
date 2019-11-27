import logging
import re
from time import perf_counter as pc
from typing import Tuple, Dict, Union

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
        encoder = lambda frame: frame
        label_columns = None
        targets = None

        # Union[List[str], TargetLabelEncoder, Dict[str, Union[List[str], TargetLabelEncoder]]]
        if isinstance(labels, list):
            targets = None
            label_columns = labels
        elif isinstance(labels, TargetLabelEncoder):
            targets = None
            encoder = labels.encode
            label_columns = labels.labels
        elif isinstance(features_and_labels.labels, Dict):
            # this is our multi model case, here we add an extra dimension to the labels array
            # TODO implement
            pass

        self.df = df
        self._features_and_labels = features_and_labels
        self._labels = label_columns
        self._targets = targets
        self._encoder = encoder

    def prediction_to_frame(self, prediction: np.ndarray, index: pd.Index = None, inclusive_labels: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        index = self.df.index if index is None else index

        # create a data frame from the prediction # FIXME multi model dimension size ..
        if len(self._labels) > 1:
            df = pd.DataFrame({l: prediction[:, i] for i, l in enumerate(self._labels)}, index=index)
        elif len(self._labels) == 1 and len( prediction.shape) > 1 and prediction.shape[1] > 1:
            df = pd.DataFrame({f'{self._labels[0]} #{i}': prediction[:, i] for i in range(prediction.shape[1])}, index=index)
        else:
            df = pd.DataFrame({self._labels[0]: prediction[:, 0] if len(prediction.shape) > 1 else prediction}, index=index)

        # assign multi level index to the predictions frame
        df.columns = pd.MultiIndex.from_arrays([[PREDICTION_COLUMN_NAME] * len(df.columns), df.columns])

        # add labels if requested
        if inclusive_labels:
            dfl = self.labels_df
            dfl.columns = pd.MultiIndex.from_arrays([[LABEL_COLUMN_NAME] * len(dfl.columns), dfl.columns])
            df = df.join(dfl, how='inner')

            # and add loss if provided
            if self._features_and_labels.loss is not None:
                dfl = self._features_and_labels.loss(None, self.df.loc[df.index]) # FIXME none vs multi model
                if isinstance(dfl, pd.Series):
                    if dfl.name is None:
                        dfl.name = LOSS_COLUMN_NAME
                    dfl = dfl.to_frame()

                dfl.columns = pd.MultiIndex.from_arrays([[LOSS_COLUMN_NAME] * len(dfl.columns), dfl.columns])
                df = df.join(dfl, how='inner')

        # add target if provided
        if self._features_and_labels.targets is not None:
            dft = self._features_and_labels.targets(None, self.df.loc[df.index]) # FIXME none vs multi model
            if isinstance(dft, pd.Series):
                if dft.name is None:
                    dft.name = TARGET_COLUMN_NAME
                dft = dft.to_frame()

            dft.columns = pd.MultiIndex.from_arrays([[TARGET_COLUMN_NAME] * len(dft.columns), dft.columns])
            df = df.join(dft, how='inner')

        return df

    @property
    def features(self) -> Tuple[pd.DataFrame, np.ndarray]:
        df = self.features_df
        x = df[self.feature_names].values if self._features_and_labels.feature_lags is None else \
            np.array([df[cols].values for cols in self.feature_names], ndmin=3).swapaxes(0, 1)

        _log.info(f"features shape: {x.shape}")
        return df, x

    @property
    def features_labels(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        df_features = self.features_df
        df_labels = self.labels_df.loc[df_features.index]
        df = self.features_df.join(df_labels).dropna()

        # features eventually are in RNN shape which is [row, time_step, feature]
        x = df[self.feature_names].values if self._features_and_labels.feature_lags is None else \
            np.array([df[cols].values for cols in self.feature_names], ndmin=3).swapaxes(0, 1)

        # labels are straight forward but eventually need to be type corrected
        # TODO if len(targets > 1) then we need an extra dimension for y array
        y = df_labels.values.astype(self._features_and_labels.label_type)

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
        df = self._encoder(self.df[self._labels]).dropna().copy()
        return df

    def __str__(self):
        return f'min required data = {self._features_and_labels.min_required_samples}'

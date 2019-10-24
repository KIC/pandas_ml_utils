import os
from functools import lru_cache

import numpy as np
import pandas as pd
import logging

from time import perf_counter as pc
from sortedcontainers import SortedDict
from typing import Type, Callable

from pandas_ml_utils.wrappers.hashable_dataframe import HashableDataFrame
from pandas_ml_utils.utils import log_with_time, ReScaler

log = logging.getLogger(__name__)


def make_backtest_data(df: pd.DataFrame,
                       features_and_labels: 'FeaturesAndLabels',
                       label_type: Type = int):
    return make_training_data(df, features_and_labels, 0, label_type)


def make_training_data(df: pd.DataFrame,
                       features_and_labels: 'FeaturesAndLabels',
                       test_size: float = 0.4,
                       label_type: Type = int,
                       seed: int = 42,
                       cache: bool = False):
    # only import if this method is needed
    from sklearn.model_selection import train_test_split

    # use only feature and label columns
    start_pc = log_with_time(lambda: log.debug("make training / test data split ..."))
    df = df[set(features_and_labels.features + features_and_labels.labels)]

    # create features and re-assign data frame with all nan rows dropped
    df_new, x = _make_features_with_cache(HashableDataFrame(df), features_and_labels) if cache else \
                _make_features(df, features_and_labels)

    # calculate the minimum required data
    min_required_data = len(df) - len(df_new) + 1

    # assign labels
    y = df_new[features_and_labels.labels].values

    # split training and test data
    start_split_pc = log_with_time(lambda: log.debug("  splitting ..."))
    x_train, x_test, y_train, y_test, index_train, index_test = \
        train_test_split(x, y, df_new.index, test_size=test_size, random_state=seed) if test_size > 0 \
            else (x, None, y, None, df_new.index, None)
    log.info(f"  splitting ... done in {pc() - start_split_pc: .2f} sec!")

    # ravel one dimensional labels
    if len(features_and_labels.labels) == 1:
        y_train = y_train.ravel().astype(label_type)
        y_test = y_test.ravel().astype(label_type) if y_test is not None else None

    log.info(f"make training / test data split ... done in {pc() - start_pc: .2f} sec!")

    # return the split
    #log.debug(f"${len(x_train)}, {len(x_test)}, {len(y_train)}, {len(y_test)}, {len(index_train)}, {len(index_test)}, {min_required_data}")
    return x_train, x_test, y_train, y_test, index_train, index_test, min_required_data


def make_forecast_data(df: pd.DataFrame, features_and_labels: 'FeaturesAndLabels'):
    return _make_features(df[features_and_labels.features], features_and_labels)


@lru_cache(maxsize=int(os.getenv('CACHE_FEATUES_AND_LABELS', '1')))
def _make_features_with_cache(df: HashableDataFrame, features_and_labels: 'FeaturesAndLabels'):
    log.info(f"no cache entry available for {hash(df), hash(features_and_labels)}")
    return _make_features(df, features_and_labels)


def _make_features(df: pd.DataFrame, features_and_labels: 'FeaturesAndLabels'):
    start_pc = log_with_time(lambda: log.debug(" make features ..."))
    feature_lags = features_and_labels.feature_lags
    feature_rescaling = features_and_labels.feature_rescaling
    features = features_and_labels.features
    lag_smoothing = features_and_labels.lag_smoothing

    # drop nan's and copy frame
    df = df.dropna().copy()

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

        # RNN shape need to be [row, time_step, feature]
        x = np.array([[[df.iloc[row][f'{feat}_{lag}']
                        for feat in features]
                       for lag in feature_lags] for row in range(len(df))],
                     ndmin=3)
    else:
        # return simple 2D arrays
        x = df[features].values

    if feature_rescaling is not None:
        for tuple, target_range in feature_rescaling.items():
            for i in range(len(x)):
                cols = [features.index(j) for j in tuple]
                if len(x.shape) == 3:
                    data = x[i, :, cols]
                    x[i,:,cols] = ReScaler((data.min(), data.max()), target_range)(data)
                elif len(x.shape) == 2:
                    data = x[i,cols]
                    x[i,cols] = ReScaler((data.min(), data.max()), target_range)(data)
                else:
                    ValueError("unknown array dimensions")

    log.info(f" make features ... done in {pc() - start_pc: .2f} sec!")
    return df, x


def reshape_rnn_as_ar(arr3d):
    if len(arr3d.shape) < 3:
        print("Data was not in RNN shape")
        return arr3d
    else:
        return arr3d.reshape(arr3d.shape[0], arr3d.shape[1] * arr3d.shape[2])


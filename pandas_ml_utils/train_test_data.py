import os
from functools import lru_cache

import numpy as np
import pandas as pd
import logging

from time import perf_counter as pc
from sortedcontainers import SortedDict
from typing import Type, Iterable, List, Callable, Dict

from wrappers.hashable_dataframe import HashableDataFrame
from pandas_ml_utils.utils import log_with_time


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
                       cache: bool = False,
                       summary_printer: Callable[[np.ndarray, np.ndarray, np.ndarray], None] = None):
    # only import if this method is needed
    from sklearn.model_selection import train_test_split

    # use only feature and label columns
    start_pc = log_with_time(lambda: log.debug("make training / test data split ..."))
    df = df[set(features_and_labels.features + features_and_labels.labels)]

    # create features and re-assign data frame with all nan rows dropped
    df_new, x, names = _make_features_with_cache(HashableDataFrame(df), features_and_labels) if cache else \
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

    # print some statistics if needed
    if summary_printer is not None:
        summary_printer(y, y_train, y_test)

    # return the split
    return x_train, x_test, y_train, y_test, index_train, index_test, min_required_data, (names, features_and_labels.labels)


def make_forecast_data(df: pd.DataFrame, features_and_labels: 'FeaturesAndLabels'):
    return _make_features(df[features_and_labels.features], features_and_labels)


@lru_cache(maxsize=int(os.getenv('CACHE_FEATUES_AND_LABELS', '1')))
def _make_features_with_cache(df: HashableDataFrame, features_and_labels: 'FeaturesAndLabels'):
    log.info(f"no cache entry available for {hash(df), hash(features_and_labels)}")
    return _make_features(df, features_and_labels)


def _make_features(df: pd.DataFrame, features_and_labels: 'FeaturesAndLabels'):
    start_pc = log_with_time(lambda: log.debug(" make features ..."))
    feature_lags = features_and_labels.feature_lags
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

        names = np.array([[f'{feat}_{lag}'
                           for feat in features]
                          for lag in feature_lags], ndmin=2)
    else:
        # return simple 2D arrays
        x = df[features].values
        names = features

    log.info(f" make features ... done in {pc() - start_pc: .2f} sec!")
    return df, x, names


def reshape_rnn_as_ar(arr3d):
    if len(arr3d.shape) < 3:
        print("Data was not in RNN shape")
        return arr3d
    else:
        return arr3d.reshape(arr3d.shape[0], arr3d.shape[1] * arr3d.shape[2])


import numpy as np
import pandas as pd

from sortedcontainers import SortedDict
from typing import Type, Iterable, List, Callable, Dict

from .training_test_data import FeaturesAndLabels


def make_backtest_data(df: pd.DataFrame,
                       features_and_labels: FeaturesAndLabels,
                       test_size: float = 0.4,
                       label_type: Type = int):
    return make_training_data(df, features_and_labels, 0, label_type)


def make_training_data(df: pd.DataFrame,
                       features_and_labels: FeaturesAndLabels,
                       test_size: float = 0.4,
                       label_type: Type = int,
                       seed = 42):
    # only import if this method is needed
    from sklearn.model_selection import train_test_split

    # use only feature and label columns
    df = df[set(features_and_labels.features + features_and_labels.labels)]

    # re-assign data frame with all nan rows dropped
    df, x, names = _make_features(df,
                                  features_and_labels.features,
                                  features_and_labels.feature_lags,
                                  features_and_labels.lag_smoothing)
    y = df[features_and_labels.labels].values

    # split training and test data
    x_train, x_test, y_train, y_test, index_train, index_test = \
        train_test_split(x, y, df.index, test_size=test_size, random_state=seed) if test_size > 0 \
            else (x, None, y, None, df.index, None)

    # ravel one dimensional labels
    if len(features_and_labels.labels) == 1:
        y_train = y_train.ravel().astype(label_type)
        y_test = y_test.ravel().astype(label_type) if y_test is not None else None

    return x_train, x_test, y_train, y_test, index_train, index_test, (names, features_and_labels.labels)


def make_forecast_data(df: pd.DataFrame, features_and_labels: FeaturesAndLabels):
    return _make_features(df[features_and_labels.features],
                          features_and_labels.features,
                          features_and_labels.feature_lags,
                          features_and_labels.lag_smoothing)


def _make_features(df: pd.DataFrame,
                   features: List[str],
                   feature_lags: Iterable[int] = None,
                   lag_smoothing: Dict[int, Callable[[pd.Series], pd.Series]] = None):

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

    return df, x, names


def reshape_rnn_as_ar(arr3d):
    if len(arr3d.shape) < 3:
        print("Data was not in RNN shape")
        return arr3d
    else:
        return arr3d.reshape(arr3d.shape[0], arr3d.shape[1] * arr3d.shape[2])


import numpy as np
import pandas as pd

from sortedcontainers import SortedDict
from typing import Type

from .training_test_data import FeaturesAndLabels


def make_training_data(df: pd.DataFrame,
                       features_and_labels: FeaturesAndLabels,
                       test_size: float = 0.4,
                       label_type: Type = int,
                       seed = 42):
    # only import if this method is needed
    from sklearn.model_selection import train_test_split

    if features_and_labels.feature_lags is not None:
        # return RNN shaped 3D arrays
        # copy features and labels
        df = df[set(features_and_labels.features + features_and_labels.labels)].copy()
        for feature in features_and_labels.features:
            feature_series = df[feature]
            smoothers = None

            # smooth out feature if requested
            if features_and_labels.lag_smoothing is not None:
                smoothers = SortedDict({lag: smoother(feature_series.to_frame())
                                        for lag, smoother in features_and_labels.lag_smoothing.items()})

            for lag in features_and_labels.feature_lags:
                # if smoothed values are applicable use smoothed values
                if smoothers is not None and len(smoothers) > 0 and smoothers.peekitem(0)[0] <= lag:
                    feature_series = smoothers.popitem(0)[1]

                # assign the lagged (eventually smoothed) feature to the features frame
                df[f'{feature}_{lag}'] = feature_series.shift(lag)

        df = df.dropna()
        index = df.index
        y = df[features_and_labels.labels].values

        # RNN shape need to be [row, time_step, feature]
        x = np.array([[[df.iloc[row][f'{feat}_{lag}']
                        for feat in features_and_labels.features]
                       for lag in features_and_labels.feature_lags] for row in range(len(df))],
                     ndmin=3)

        names = (np.array([[f'{feat}_{lag}'
                            for feat in features_and_labels.features]
                           for lag in features_and_labels.feature_lags], ndmin=2),
                 features_and_labels.labels)
    else:
        # return simple 2D arrays
        df = df.dropna()
        x = df[features_and_labels.features].values
        y = df[features_and_labels.labels].values
        index = df.index
        names = (features_and_labels.features, features_and_labels.labels)

    x_train, x_test, y_train, y_test, index_train, index_test = \
        train_test_split(x, y, index, test_size=test_size, random_state=seed) if test_size > 0 else (x, None, y, None, df.index, None)

    # ravel one dimensional labels
    if len(features_and_labels.labels) == 1:
        y_train = y_train.ravel().astype(label_type)
        y_test = y_test.ravel().astype(label_type) if y_test is not None else None

    return x_train, x_test, y_train, y_test, index_train, index_test, names


def reshape_rnn_as_ar(arr3d):
    if len(arr3d.shape) < 3:
        print("Data was not in RNN shape")
        return arr3d
    else:
        return arr3d.reshape(arr3d.shape[0], arr3d.shape[1] * arr3d.shape[2])


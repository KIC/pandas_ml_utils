import re
import numpy as np
import pandas as pd

from typing import Dict, Union, Callable, List, Iterable
from sortedcontainers import SortedDict
from .training_test_data import TrainTestData


def add_apply(df, **kwargs: Callable[[pd.DataFrame], Union[pd.Series, pd.DataFrame]]):
    df2 = pd.DataFrame()
    for k, v in kwargs.items():
        df2[k] = v(df)

    return df.join(df2)


def shift_inplace(df, **kwargs: int):
    for k, v in kwargs.items():
        df[k] = df[k].shift(v)

    return df


def drop_re(df, *args: str):
    drop_list = []

    for regex in args:
        drop_list.extend(list(filter(re.compile(regex).match, df.columns)))

    return df.drop(drop_list, axis=1)


def extend_forecast(df, periods: int):
    df_ext = pd.DataFrame(index=pd.date_range(df.index[-1], periods=periods+1, closed='right'))
    return pd.concat([df, df_ext], axis=0, sort=True)


def make_training_data(df: pd.DataFrame,
                       features: List[str],
                       labels: List[str],
                       test_size: float = 0.4,
                       feature_lags: Iterable[int] = None,
                       lag_smoothing: Dict[int, Callable[[pd.Series], pd.Series]] = None,
                       seed = 42):
    # only import if this method is needed
    from sklearn.model_selection import train_test_split

    if feature_lags is not None:
        # return RNN shaped 3D arrays
        # copy features and labels
        df = df[set(features + labels)].copy()
        for feature in features:
            feature_series = df[feature]
            smoothers = None

            # smooth out feature if requested
            if lag_smoothing is not None:
                smoothers = SortedDict({lag: smoother(feature_series.to_frame()) for lag, smoother in lag_smoothing.items()})

            for lag in feature_lags:
                # if smoothed values are applicable use smoothed values
                if smoothers is not None and len(smoothers) > 0 and smoothers.peekitem(0)[0] <= lag:
                    feature_series = smoothers.popitem(0)[1]

                # assign the lagged (eventually smoothed) feature to the features frame
                df[f'{feature}_{lag}'] = feature_series.shift(lag)

        df = df.dropna()
        index = df.index
        y = df[labels].values

        # RNN shape need to be [row, time_step, feature]
        x = np.array([[[df.iloc[row][f'{feat}_{lag}'] for feat in features] for lag in feature_lags] for row in range(len(df))],
                     ndmin=3)

        names = (np.array([[f'{feat}_{lag}' for feat in features] for lag in feature_lags], ndmin=2), labels)
    else:
        # return simple 2D arrays
        df = df.dropna()
        x = df[features].values
        y = df[labels].values
        index = df.index
        names = (features, labels)

    x_train, x_test, y_train, y_test, index_train, index_test = \
        train_test_split(x, y, index, test_size=test_size, random_state=seed) if test_size > 0 else (x, None, y, None, df.index, None)

    # ravel one dimensional labels
    if len(labels) == 1:
        y_train = y_train.ravel()
        y_test = y_test.ravel() if y_test is not None else None

    return TrainTestData(x_train, x_test, y_train, y_test, names)

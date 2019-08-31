import re
import numpy as np
import pandas as pd

from typing import Dict, Union, Callable, List, Iterable


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
        # copy features and labels
        df = df[features + labels]
        for feature in features:
            for lag in feature_lags:
                # TODO if lag > x then use averaged values
                df[f'{feature}_{lag}'] = df[feature].shift(lag)
                pass

        df = df.dropna()
        y = df[labels].values

        # RNN shape need to be [row, time_step, feature]
        x = np.array([[[df.iloc[row][f'{feat}_{lag}'] for feat in features] for lag in feature_lags] for row in range(len(df))],
                     ndmin=3)
    else:
        df = df.dropna()
        x = df[features].values
        y = df[labels].values

    return train_test_split(x, y, test_size=test_size, random_state=seed) if test_size > 0 else (x, None, y, None)

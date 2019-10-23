import logging
import re
from typing import Union, Callable, List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


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


def drop_zero_or_nan(df, columns: List[str] = None, epsilon=1e-10):
    df = df.copy()

    # replace all zeros by nan
    for col in columns if columns is not None else df.columns:
        df[df[col].abs() < epsilon] = np.NaN

    # then drop all nan
    return df.dropna()


def extend_forecast(df, periods: int):
    df_ext = pd.DataFrame(index=pd.date_range(df.index[-1], periods=periods+1, closed='right'))
    return pd.concat([df, df_ext], axis=0, sort=True)

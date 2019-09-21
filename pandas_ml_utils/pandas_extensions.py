import logging
import re
from typing import Union, Callable

import pandas as pd

from wrappers.hashable_dataframe import HashableDataFrame

log = logging.getLogger(__name__)


def hashable(df):
    return HashableDataFrame(df)


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

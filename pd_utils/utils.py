import re
import pandas as pd

from typing import Dict, Union, Callable


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


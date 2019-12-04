# make imports private to allow star import
import re as _re
import numpy as _np
import pandas as _pd


def add_apply(df, **kwargs):
    df2 = _pd.DataFrame()
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
        drop_list.extend(list(filter(_re.compile(regex).match, df.columns)))

    return df.drop(drop_list, axis=1)


def drop_zero_or_nan(df, columns = None, epsilon=1e-10):
    df = df.copy()

    # replace all zeros by nan
    for col in columns if columns is not None else df.columns:
        df[df[col].abs() < epsilon] = _np.NaN

    # then drop all nan
    return df.dropna()


def extend_forecast(df, periods: int):
    df_ext = _pd.DataFrame(index=_pd.date_range(df.index[-1], periods=periods + 1, closed='right'))
    return _pd.concat([df, df_ext], axis=0, sort=True)


def inner_join(df, join: _pd.DataFrame, prefix: str = ''):
    return _pd.merge(df, join.add_prefix(prefix), left_index=True, right_index=True, how='inner', sort=True)

import uuid
from functools import lru_cache
from typing import Callable, Union

import pandas as pd

from pandas_ml_utils.model.fitting.fitter import fit, predict, backtest


class LazyDataFrame(object):

    def __init__(self, df: pd.DataFrame, **kwargs: Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]]) -> None:
        self.hash = uuid.uuid4()
        self.df: pd.DataFrame = df
        self.kwargs = kwargs

    @property
    def columns(self):
        return self.df.columns.tolist() + list(self.kwargs.keys())

    @property
    def index(self):
        return self.df.index

    def fit(self, *args, **kwargs):
        return fit(self, *args, **kwargs)

    def backtest(self, *args, **kwargs):
        return backtest(self, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return predict(self, *args, **kwargs)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item: str):
        return self.to_dataframe()[item]

    def __setitem__(self, key: str, value: Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]]):
        self.hash = uuid.uuid4()
        if callable(value):
            self.kwargs[key] = value(self.df)
        else:
            self.df[key] = value

    def __getattr__(self, item):
        if isinstance(item, str) and item.startswith("__") and item.endswith("__"):
            # do not allow private private items
            raise AttributeError(f'{type(self)} has not attr {item}')
        else:
            return self.to_dataframe().__getattr__(item)

    def __contains__(self, key):
        return key in self.to_dataframe()

    def __hash__(self):
        return int(self.hash)

    def __eq__(self, other):
        return self.hash == other.hash if isinstance(other, LazyDataFrame) else False

    def __str__(self):
        return f'{self.hash}, {self.kwargs.keys()}'

    def __deepcopy__(self, memodict={}):
        return LazyDataFrame(self.df, **self.kwargs)

    def with_dataframe(self, df: pd.DataFrame):
        return LazyDataFrame(df, **self.kwargs)

    @lru_cache(maxsize=1)
    def to_dataframe(self) -> pd.DataFrame:
        df = self.df.copy()
        for key, calculation in self.kwargs.items():
            column = calculation(df)
            if isinstance(column, pd.DataFrame):
                df = df.join(column.add_prefix(f'{key}_'))
            else:
                df[key] = column

        return df
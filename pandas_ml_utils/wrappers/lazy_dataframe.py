import uuid
import pandas as pd
from typing import Callable, Union


class LazyDataFrame(object):

    def __init__(self, df: pd.DataFrame, **kwargs: Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]]) -> None:
        self.hash = uuid.uuid4()
        self.df: pd.DataFrame = df
        self.kwargs = kwargs

    def __getitem__(self, item: str):
        if isinstance(item, list):
            df = self.df[[value for value in item if value in self.df.columns]]
            for key in item:
                if key in self.kwargs:
                    res = self.kwargs[key](self.df)
                    if isinstance(res, pd.Series):
                        res.name = key
                        df = df.join(res)
                    elif isinstance(res, pd.DataFrame):
                        df = df.join(res.add_prefix(f'{key}_'))

            return df
        else:
            if item in self.df:
                return self.df[item]
            elif item in self.kwargs:
                return self.kwargs[item](self.df)
            else:
                raise ValueError(f"invalid item {item}")

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
        return key in self.df or key in self.kwargs

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

    def to_dataframe(self):
        df = self.df.copy()
        for key, calculation in self.kwargs.items():
            column = calculation(df)
            if isinstance(column, pd.DataFrame):
                df = df.join(column.add_prefix(f'{key}_'))
            else:
                df[key] = column

        return df
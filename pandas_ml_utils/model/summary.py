import pandas as pd
from typing import Dict


class Summary(object):

    def __init__(self, df: pd.DataFrame, **kwargs):
        self._df = df
        self.kwargs = kwargs

    @property
    def df(self):
        return self._df

    def _repr_html_(self):
        pass

    def _html_(self, width: str = '100%'):
        pass

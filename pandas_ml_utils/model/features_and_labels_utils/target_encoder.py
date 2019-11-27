import pandas as pd
import numpy as np
from typing import Iterable, List

from pandas_ml_utils.utils.functions import one_hot


class TargetLabelEncoder(object):

    @property
    def labels(self) -> List[str]:
        pass

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class OneHotEncodedTargets(TargetLabelEncoder):
    """
    Turns a continuous variable into a discrete one using buckets which finally get one-hot encoded

    Example usage:

        df["label"] = df["a"] / df["b"] - 1
        df.fit_classifier(Model(FeaturesAndLabels(["feature"], ["label"],
                                targets=OneHotEncodedTargets(range(-5, 5), False))
    """

    def __init__(self, label: str, rrange: Iterable, closed=False):
        super().__init__()
        borders = list(rrange)

        if closed:
            self.buckets = pd.IntervalIndex.from_tuples([(borders[r], borders[r + 1]) for r in range(len(borders) - 1)])
        else:
            self.buckets = pd.IntervalIndex.from_tuples(
                [(-float("inf") if r == 0 else borders[r], float("inf") if r == len(borders) - 2 else borders[r + 1]) for r in
                 range(len(borders) - 1)])

        self.label = label
        self.number_of_categories = len(self.buckets)

    @property
    def labels(self) -> List[str]:
        return [self.label]

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self.label
        buckets = pd.cut(df[col], self.buckets)
        indexes = buckets.cat.codes.values

        one_hot_matrix = np.array([one_hot(i, self.number_of_categories) for i in indexes]).T
        one_hot_categories = pd.DataFrame({f'{col} #{i}': v for i, v in enumerate(one_hot_matrix)}, index=df.index)

        return one_hot_categories


import pandas as pd
import numpy as np
from typing import Iterable


class TargetLabelEncoder(object):

    @property
    def targets(self):
        return None

    @property
    def labels(self):
        return None


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

    def targets(self):
        return None

    def labels(self):
        return None

    """
    def __init__(self, series, buckets):
        def naming(name, category):
            return f"{name if isinstance(buckets, str) else category}"

        bucketers = list(range(buckets) if isinstance(buckets, int) else buckets)
        buckets = pd.cut(series, bins=len(bucketers) if bucketers else buckets)

        indexes = buckets.cat.codes.values
        categories = buckets.cat.categories.tolist()
        number_of_categories = len(categories)

        names = [naming(bucketers[i], c) for i, c in enumerate(categories)]

        one_hot_matrix = np.array([one_hot(i, number_of_categories) for i in indexes]).T
        target_matrix = np.array([series.apply(lambda x: f(categories[i], x) if callable(f) else f).values
                                  for i, f in enumerate(bucketers)])

        self.one_hot_categories = {names[i]: v for i, v in enumerate(one_hot_matrix)}
        self.one_hot_targets = {f"{names[i]}_target": v for i, v in enumerate(target_matrix)}
    """


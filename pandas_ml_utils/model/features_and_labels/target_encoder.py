import pandas as pd
import numpy as np
from typing import Iterable, List, Dict

from pandas_ml_utils.utils.functions import one_hot


class TargetLabelEncoder(object):

    @property
    def labels_source_columns(self) -> List[str]:
        pass

    @property
    def encoded_labels_columns(self) -> List[str]:
        pass

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def __len__(self):
        1


class IdentityEncoder(TargetLabelEncoder):

    def __init__(self, target_labels: List[str]):
        super().__init__()
        self.target_labels = target_labels

    @property
    def labels_source_columns(self) -> List[str]:
        return self.target_labels

    @property
    def encoded_labels_columns(self) -> List[str]:
        return self.target_labels

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.target_labels]

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def __len__(self):
        return len(self.target_labels)


class MultipleTargetEncodingWrapper(TargetLabelEncoder):

    def __init__(self, target_labels: Dict[str, TargetLabelEncoder]):
        super().__init__()
        self.target_labels = target_labels

    @property
    def labels_source_columns(self) -> List[str]:
        return [l for enc in self.target_labels.values() for l in enc.labels_source_columns]

    @property
    def encoded_labels_columns(self) -> List[str]:
        pass

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df_labels = pd.DataFrame({}, index=df.index)
        for target, enc in self.target_labels.items():
            df_labels = df_labels.join(enc.encode(df), how='inner')

        return df_labels

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        # FIXME
        pass

    def __len__(self):
        sum([len(enc) for enc in self.target_labels.values()])


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
    def labels_source_columns(self) -> List[str]:
        return [self.label]

    @property
    def encoded_labels_columns(self) -> List[str]:
        #return [str(11) if isinstance(cat, pd._libs.interval.Interval) else str(cat) for cat in self.buckets]
        return [str(cat) for cat in self.buckets]

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self.label
        buckets = pd.cut(df[col], self.buckets)
        indexes = buckets.cat.codes.values

        one_hot_matrix = np.array([one_hot(i, self.number_of_categories) for i in indexes]).T
        one_hot_categories = pd.DataFrame({f'{col} #{i}': v for i, v in enumerate(one_hot_matrix)}, index=df.index)

        return one_hot_categories

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        # FIXME
        pass

    def __len__(self):
        return len(self.buckets)
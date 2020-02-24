from copy import deepcopy

import pandas as pd
import numpy as np
from typing import Iterable, List, Dict, Union, Callable

from pandas_ml_utils.utils.functions import one_hot, call_callable_dynamic_args, join_kwargs


class TargetLabelEncoder(object):

    def __init__(self):
        self.kwargs = {}

    @property
    def labels_source_columns(self) -> List[str]:
        pass

    @property
    def encoded_labels_columns(self) -> List[str]:
        pass

    def encode(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def with_kwargs(self, **kwargs):
        copy = deepcopy(self)
        copy.kwargs = join_kwargs(copy.kwargs, kwargs)
        return copy

    def __len__(self):
        return 1


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

    def encode(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
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
        return [l for enc in self.target_labels.values() for l in enc.encoded_labels_columns]

    def encode(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df_labels = pd.DataFrame({}, index=df.index)
        for target, enc in self.target_labels.items():
            df_labels = df_labels.join(enc.encode(df), how='inner')

        return df_labels

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        df_labels = pd.DataFrame({}, index=df.index)
        for target, enc in self.target_labels.items():
            df_labels = df_labels.join(enc.decode(df), how='inner', rsuffix=f'_{target}')

        return df_labels

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
        return [str(cat) for cat in self.buckets]

    def encode(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        col = self.label
        buckets = pd.cut(df[col], self.buckets)
        indexes = buckets.cat.codes.values

        one_hot_matrix = np.array([one_hot(i, self.number_of_categories) for i in indexes]).T
        one_hot_categories = pd.DataFrame({f'{col} #{i}': v for i, v in enumerate(one_hot_matrix)}, index=df.index)

        return one_hot_categories

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(lambda r: self.buckets[np.argmax(r)], raw=True, axis=1)

    def __len__(self):
        return len(self.buckets)


class OneHotEncodedDiscrete(TargetLabelEncoder):

    def __init__(self,
                 label: str,
                 nr_of_categories: int,
                 pre_processor: Callable[[pd.DataFrame], pd.Series] = None,
                 **kwargs):
        super().__init__()
        self.label = label
        self.nr_of_categories = nr_of_categories
        self.pre_processor = pre_processor
        self.kwargs = kwargs

    @property
    def labels_source_columns(self) -> List[str]:
        return [self.label]

    @property
    def encoded_labels_columns(self) -> List[str]:
        return [f'{self.label}_{i}' for i in range(self.nr_of_categories)]

    def encode(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # eventually pre-process data
        joined_kwargs = join_kwargs(self.kwargs, kwargs)
        sf = (call_callable_dynamic_args(self.pre_processor, df, **joined_kwargs) if self.pre_processor else df)

        # extract single series for one hot encoding
        if isinstance(sf, pd.Series):
            s = sf.rename(self.label)
        else:
            s = sf[self.label]

        # one hot encode and return
        return s.to_frame().apply(lambda r: one_hot(r.values.sum(), self.nr_of_categories), axis=1, result_type='expand')

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(lambda r: np.argmax(r), raw=True, axis=1)

    def __len__(self):
        return self.nr_of_categories

import re
from typing import Union, Callable, Tuple

import numpy as np
import pandas as pd

from pd_utils.make_train_test_data import make_training_data
from .training_test_data import FeaturesAndLabels, Model, Classification
from .make_train_test_data import reshape_rnn_as_ar


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


def fit_skit_classifier(df: pd.DataFrame,
                        features_and_labels: FeaturesAndLabels,
                        sklearn_model: Model,
                        test_size: float = 0.4,
                        test_validate_split_seed = 42) -> Tuple[Model, Classification]:
    return fit_classifier(df,
                          features_and_labels,
                          lambda: sklearn_model,
                          lambda model, x, y, x_validate, y_validate: model.fit(reshape_rnn_as_ar(x), y),
                          lambda model, x: model.predict_proba(reshape_rnn_as_ar(x))[:, 1],
                          test_size,
                          test_validate_split_seed)


def fit_classifier(df: pd.DataFrame,
                   features_and_labels: FeaturesAndLabels,
                   model_provider: Callable[[], Model],
                   model_fitter: Callable[[Model, np.ndarray, np.ndarray, np.ndarray, np.ndarray], Model],
                   model_predictor: Callable[[Model, np.ndarray], np.ndarray],
                   test_size: float = 0.4,
                   test_validate_split_seed = 42) -> Tuple[Model, Classification]:
    x_train, x_test, y_train, y_test, index_train, index_test, names = \
        make_training_data(df, features_and_labels, test_size, test_validate_split_seed)

    model = model_provider()
    res = model_fitter(model, x_train, y_train, x_test, y_test)

    if isinstance(res, type(model)):
        model = res

    #FIXME train_confusion = confusion_matrix(self.y_train, model_predictor(model, self.x_train) > probability_cutoff)
    #FIXME test_confusion = confusion_matrix(self.y_test, model_predictor(model, self.x_test) > probability_cutoff)

    return model, None #FIXME , train_confusion, test_confusion


def classify(df: pd.DataFrame,
             features_and_labels: FeaturesAndLabels,
             model_predictor: Callable[[np.ndarray], np.ndarray]) -> Classification:
    x, _, _, _, index_train, _, names = make_training_data(df, features_and_labels, 0, 0)

    return model_predictor(x)

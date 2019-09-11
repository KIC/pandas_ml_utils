import re
from typing import Union, Callable, Tuple, List

import numpy as np
import pandas as pd

from pd_utils.make_train_test_data import make_training_data, make_forecast_data
from .training_test_data import FeaturesAndLabels, Model, ClassificationSummary, Fit
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
                        skitlearn_model: Model,
                        test_size: float = 0.4,
                        test_validate_split_seed = 42) -> Tuple[Model, ClassificationSummary, ClassificationSummary]:
    return fit_classifier(df,
                          features_and_labels,
                          lambda: skitlearn_model,
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
                   test_validate_split_seed = 42) -> Tuple[Model, ClassificationSummary, ClassificationSummary]:
    x_train, x_test, y_train, y_test, index_train, index_test, names = \
        make_training_data(df, features_and_labels, test_size, int, test_validate_split_seed)

    model = model_provider()
    res = model_fitter(model, x_train, y_train, x_test, y_test)

    if isinstance(res, type(model)):
        model = res

    pc = features_and_labels.probability_cutoff
    training_classification = ClassificationSummary(y_train, model_predictor(model, x_train), index_train, df[features_and_labels.loss_column], pc)
    test_classification = ClassificationSummary(y_test, model_predictor(model, x_test), index_test, df[features_and_labels.loss_column], pc)

    return Fit(model, training_classification, test_classification)


def skit_backtest(df: pd.DataFrame,
                  features_and_labels: FeaturesAndLabels,
                  model: Model) -> ClassificationSummary:
    return backtest(df,
                    features_and_labels,
                    lambda x: model.predict_proba(reshape_rnn_as_ar(x))[:, 1])


def backtest(df: pd.DataFrame,
             features_and_labels: FeaturesAndLabels,
             model_predictor: Callable[[np.ndarray], np.ndarray]) -> ClassificationSummary:

    # make training and test data with no 0 test data fraction
    x, _, y, _, index, _, names = make_training_data(df, features_and_labels, 0, int)

    # precidict probabilities
    y_hat = model_predictor(x)

    return ClassificationSummary(y, y_hat, index, df[features_and_labels.loss_column], features_and_labels.probability_cutoff)


def skit_classify(df: pd.DataFrame,
                  features_and_labels: FeaturesAndLabels,
                  model: Model) -> pd.DataFrame:
    return classify(df,
                    features_and_labels,
                    lambda x: model.predict_proba(reshape_rnn_as_ar(x))[:, 1])


def classify(df: pd.DataFrame,
             features_and_labels: FeaturesAndLabels,
             model_predictor: Callable[[np.ndarray], np.ndarray]) -> pd.DataFrame:

    # first save target columns
    target = df[features_and_labels.target_columns] if features_and_labels.target_columns is not None else None

    # then re assign data frame with features only
    dff, x, _ = make_forecast_data(df, features_and_labels)

    # predict on features
    prediction = model_predictor(x)
    pc = features_and_labels.probability_cutoff

    # return result
    dff["prediction"] = prediction > pc
    dff["prediction_proba"] = prediction
    dff["target"] = target
    return dff

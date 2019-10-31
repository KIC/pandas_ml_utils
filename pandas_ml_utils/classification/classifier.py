import logging
from typing import Callable, Tuple, Dict

import numpy as np
import pandas as pd

from ..classification.summary import ClassificationSummary
from ..model.fit import Fit
from ..model.fitter import _fit, _backtest, _predict
from ..model.models import Model
from ..constants import *

log = logging.getLogger(__name__)


def fit_classifier(df: pd.DataFrame,
                   model_provider: Callable[[int], Model],
                   test_size: float = 0.4,
                   cross_validation: Tuple[int, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
                   cache_feature_matrix: bool = False,
                   test_validate_split_seed = 42,
                   hyper_parameter_space: Dict = None,
                   ) -> Fit:

    # maybe later we can check if only the cut off changed and then skip the fitting step
    model, (df_train, df_test), trails = _fit(df,
                                              model_provider,
                                              test_size = test_size,
                                              cross_validation = cross_validation,
                                              cache_feature_matrix = cache_feature_matrix,
                                              test_validate_split_seed = test_validate_split_seed,
                                              hyper_parameter_space = hyper_parameter_space)

    # assemble the result objects
    features_and_labels = model.features_and_labels
    cutoff = model[("probability_cutoff", 0.5)]

    # convert probabilities into classes
    df_train = _convert_probabilities(df_train, cutoff)
    df_test = _convert_probabilities(df_test, cutoff)

    loss = __get_loss(df, features_and_labels)
    training_classification = ClassificationSummary(train[1], prediction[0], index[0], loss, cutoff)
    test_classification = ClassificationSummary(test[1], prediction[1], index[1], loss, cutoff)
    return Fit(model, training_classification, test_classification, trails)


def backtest_classifier(df: pd.DataFrame, model: Model) -> ClassificationSummary:
    df = _backtest(df, model)

    return None #FIXME ClassificationSummary(y, y_hat, index, loss, model[("probability_cutoff", 0.5)])


# FIXME onsolete
def __get_loss(df, features_and_labels):
    goals = features_and_labels.get_goals()
    if len(goals) > 1:
        # FIXME we actually want to have the possibility of multiple losses ... also to plot
        log.warning("currently we only support one *loss* column, we just pick the first one")

    loss, _ = next(iter(goals.values()))
    return \
        df[loss] if loss is not None and loss in df.columns else pd.Series(-1 if loss is None else loss, index=df.index)


def classify(df: pd.DataFrame, model: Model, tail: int = None) -> pd.DataFrame:
    dff = _predict(df, model, tail)

    # return result
    for column in dff.columns:
        if column.startswith(PREDICTION_COLUMN_NAME):
            dff[f"{column}_proba"] = dff[column]
            dff[column] = dff[f"{column}_proba"] > model[("probability_cutoff", 0.5)]

    return dff


def _convert_probabilities(df: pd.DataFrame, cut_off: float = 0.5) -> pd.DataFrame:

    # return result
    for column in df.columns:
        if column[1] == PREDICTION_COLUMN_NAME:
            probability_column = (column[0], column[1], f"{column[2]}{PROBABILITY_POSTFIX}")
            df[probability_column] = df[column]
            df[column] = df[probability_column] > cut_off

    return df.sort_index(axis=1)

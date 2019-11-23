import logging
from typing import Callable, Tuple, Dict

import numpy as np
import pandas as pd

from ..classification.summary import ClassificationSummary
from ..model.fit import Fit
from ..model.fitter import _fit, _backtest, _predict
from ..model.models import Model
from ..constants import *

_log = logging.getLogger(__name__)


def fit_classifier(df: pd.DataFrame,
                   model_provider: Callable[[int], Model],
                   test_size: float = 0.4,
                   cross_validation: Tuple[int, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
                   cache_feature_matrix: bool = False,
                   test_validate_split_seed = 42,
                   hyper_parameter_space: Dict = None,
                   ) -> Fit:
    """

    :param df: the DataFrame you apply this function to
    :param model_provider: a callable which provides a new :class:`.Model` instance i.e. for each hyper parameter if
                           hyper parameter tuning is enforced. Usually all the Model subclasses implement __call__
                           thus they are a provider of itself
    :param test_size: the fraction [0, 1] of samples which are used for a test set
    :param cross_validation: tuple of number of epochs for each fold provider and a cross validation provider
    :param cache_feature_matrix: whether to cache an expensive generation of feature matrices between fit calls
    :param test_validate_split_seed: seed if train, test split needs to be reproduceable
    :param hyper_parameter_space: space of hyper parameters passed as kwargs to your model provider
    :return: returns a :class:`.Fit` object
    """
    # maybe later we can check if only the cut off changed and then skip the fitting step
    model, (df_train, df_test), trails = _fit(df,
                                              model_provider,
                                              test_size = test_size,
                                              cross_validation = cross_validation,
                                              cache_feature_matrix = cache_feature_matrix,
                                              test_validate_split_seed = test_validate_split_seed,
                                              hyper_parameter_space = hyper_parameter_space)

    # assemble the result objects
    cutoff = model[("probability_cutoff", 0.5)]

    # convert probabilities into classes
    df_train = _convert_probabilities(df_train, cutoff)
    df_test = _convert_probabilities(df_test, cutoff)

    training_classification = ClassificationSummary(df_train, cutoff)
    test_classification = ClassificationSummary(df_test, cutoff)
    return Fit(model, training_classification, test_classification, trails)


def backtest_classifier(df: pd.DataFrame, model: Model) -> ClassificationSummary:
    df = _backtest(df, model)
    df_sumary = _convert_probabilities(df)
    return ClassificationSummary(df_sumary, model[("probability_cutoff", 0.5)])


def classify(df: pd.DataFrame, model: Model, tail: int = None) -> pd.DataFrame:
    dff = _predict(df, model, tail)
    return _convert_probabilities(dff, model[("probability_cutoff", 0.5)])


def _convert_probabilities(df: pd.DataFrame, cut_off: float = 0.5) -> pd.DataFrame:

    # return result
    for column in df.columns:
        if column[1] == PREDICTION_COLUMN_NAME:
            probability_column = (column[0], column[1], f"{column[2]}{PROBABILITY_POSTFIX}")
            df[probability_column] = df[column]
            df[column] = df[probability_column] > cut_off

    return df.sort_index(axis=1)

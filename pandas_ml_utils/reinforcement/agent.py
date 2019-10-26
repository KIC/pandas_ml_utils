import logging
from typing import Callable, Tuple, Dict

import numpy as np
import pandas as pd

from ..model.fit import Fit
from ..model.fitter import _fit, _predict
from ..model.models import Model
from ..train_test_data import make_training_data
from .summary import ReinforcementSummary

log = logging.getLogger(__name__)


def fit_agent(df: pd.DataFrame,
              model_provider: Callable[[int], Model],
              test_size: float = 0.4,
              cross_validation: Tuple[int, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
              cache_feature_matrix: bool = False,
              test_validate_split_seed = 42,
              hyper_parameter_space: Dict = None,
              ) -> Fit:

    model, train, test, index, trails = _fit(df,
                                             model_provider,
                                             test_size = test_size,
                                             cross_validation = cross_validation,
                                             cache_feature_matrix = cache_feature_matrix,
                                             test_validate_split_seed = test_validate_split_seed,
                                             hyper_parameter_space=hyper_parameter_space)

    train_targets = df[model.features_and_labels.target_columns].loc[index[0]]
    test_targets = df[model.features_and_labels.target_columns].loc[index[1]]

    training_classification = ReinforcementSummary(train_targets, model.history[0])
    test_classification = ReinforcementSummary(test_targets, model.history[1])
    return Fit(model, training_classification, test_classification, trails)


def backtest_agent(df: pd.DataFrame, model: Model) -> ReinforcementSummary:
    features_and_labels = model.features_and_labels

    # make training and test data with no 0 test data fraction
    x, _, y, _, index, _, _ = make_training_data(df, features_and_labels, 0, int)

    targets = df[model.features_and_labels.target_columns]
    back_test_history = model.back_test(index, x, y)

    return ReinforcementSummary(targets, back_test_history)


def agent_take_action(df: pd.DataFrame, model: Model, tail: int = None) -> pd.DataFrame:
    dff = _predict(df, model, tail)
    return dff

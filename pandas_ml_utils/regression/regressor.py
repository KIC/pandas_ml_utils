
import logging
from typing import Callable, Tuple, Dict

import numpy as np
import pandas as pd

from .summary import RegressionSummary
from ..model.fit import Fit
from ..model.fitter import _fit, _backtest, _predict
from ..model.models import Model
from ..error.functions import mse as _mse
from ..constants import *

_log = logging.getLogger(__name__)


def fit_regressor(df: pd.DataFrame,
                  model_provider: Callable[[int], Model],
                  test_size: float = 0.4,
                  cross_validation: Tuple[int, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
                  cache_feature_matrix: bool = False,
                  test_validate_split_seed = 42,
                  hyper_parameter_space: Dict = None,
                  ) -> Fit:

    model, (df_train, df_test), trails  = _fit(df,
                                               model_provider,
                                               test_size = test_size,
                                               cross_validation = cross_validation,
                                               cache_feature_matrix = cache_feature_matrix,
                                               test_validate_split_seed = test_validate_split_seed,
                                               hyper_parameter_space=hyper_parameter_space)

    training_summary = RegressionSummary(df_train)
    test_summary = RegressionSummary(df_test)
    return Fit(model, training_summary, test_summary, trails)


def backtest_regressor(df: pd.DataFrame, model: Model) -> None:
    df = _backtest(df, model)
    return RegressionSummary(df)


def regress(df: pd.DataFrame, model: Model, tail: int = None) -> pd.DataFrame:
    return _predict(df, model, tail)

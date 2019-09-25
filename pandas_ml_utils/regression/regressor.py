
import logging
from typing import Callable

import numpy as np
import pandas as pd

from ..model.fit import Fit
from ..model.fitter import _fit, _backtest, _predict
from ..model.models import Model
from ..error.functions import mse as _mse
log = logging.getLogger(__name__)


# FIXME
def fit_regressor(df: pd.DataFrame,
                  model_provider: Callable[[int], Model],
                  test_size: float = 0.4,
                  number_of_cross_validation_splits: int = None,  # FIXME make some kind of lambda
                  cache_feature_matrix: bool = False,
                  test_validate_split_seed = 42,
                  summary_printer: Callable[[np.ndarray, np.ndarray, np.ndarray], None] = None
                  ) -> Fit:

    model, train, test, index = _fit(df,
                                     model_provider,
                                     test_size = test_size,
                                     number_of_cross_validation_splits = number_of_cross_validation_splits,
                                     cache_feature_matrix = cache_feature_matrix,
                                     test_validate_split_seed = test_validate_split_seed,
                                     summary_printer = summary_printer)

    # assemble the result objects
    features_and_labels = model.features_and_labels

    # loss = df[features_and_labels.loss_column] if features_and_labels.loss_column is not None else None
    # training_classification = ClassificationSummary(train[1], model.predict(train[0]), index[0], loss, cutoff)
    # test_classification = ClassificationSummary(test[1], model.predict(test[0]), index[1], loss, cutoff)
    # return Fit(model, training_classification, test_classification)

    # TODO
    return Fit(model, None, None)


# FIXME
def backtest_regressor(df: pd.DataFrame, model: Model) -> None:
    x, y, y_hat, index = _backtest(df, model)

    features_and_labels = model.features_and_labels
    loss = df[features_and_labels.loss_column if features_and_labels.loss_column is not None else []]
    # return ClassificationSummary(y, y_hat, index, loss, model[("probability_cutoff", 0.5)])
    return None


def regress(df: pd.DataFrame, model: Model, ) -> pd.DataFrame:
    dff = _predict(df, model, None)

    # get labels calculate error
    error_function = model[("error", _mse)]
    dff["error"] = error_function(df[model.features_and_labels.labels], dff[[col for col in dff if col.startswith('prediction')]])

    return dff

import logging
from time import perf_counter
from typing import Callable, Tuple

import numpy as np
import pandas as pd

from ..train_test_data import make_training_data, make_forecast_data
from ..utils import log_with_time
from ..model.models import Model
from ..classification.summary import ClassificationSummary

log = logging.getLogger(__name__)


def _fit(df: pd.DataFrame,
        model_provider: Callable[[int], Model],
        test_size: float = 0.4,
        cross_validation: Tuple[int, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None, # FIXME make this a tuple and use [0] as loop
        cache_feature_matrix: bool = False,
        test_validate_split_seed = 42,
        summary_printer: Callable[[np.ndarray, np.ndarray, np.ndarray], None] = None # TODO lets provide a summary provider for the result like lambda model, df, fal, x, y, y_hat: ClassificationSummary(...)
        ) -> Tuple[Model, Tuple, Tuple, Tuple]:
    # get a new model
    model = model_provider()
    features_and_labels = model.features_and_labels

    # make training and test data sets
    x_train, x_test, y_train, y_test, index_train, index_test, min_required_data = \
        make_training_data(df,
                           features_and_labels,
                           test_size,
                           int,
                           test_validate_split_seed,
                           cache=cache_feature_matrix,
                           summary_printer=summary_printer)

    log.info(f"create model (min required data = {min_required_data}")
    model.min_required_data = min_required_data

    # fit the model
    start_performance_count = log_with_time(lambda: log.info("fit model"))
    if cross_validation is not None and isinstance(cross_validation, Tuple) and callable(cross_validation[1]):
        for fold_epoch in range(cross_validation[0]):
            # cross validation, make sure we re-shuffle every fold_epoch
            for f, (train_idx, test_idx) in enumerate(cross_validation[1](x_train, y_train)):
                log.info(f'fit fold {f}')
                model.fit(x_train[train_idx], y_train[train_idx], x_train[test_idx], y_train[test_idx])
    else:
        # fit without cross validation
        model.fit(x_train, y_train, x_test, y_test)

    log.info(f"fitting model done in {perf_counter() - start_performance_count: .2f} sec!")
    return model, (x_train, y_train), (x_test, y_test), (index_train, index_test)


def _backtest(df: pd.DataFrame, model: Model) -> ClassificationSummary:
    features_and_labels = model.features_and_labels

    # make training and test data with no 0 test data fraction
    x, _, y, _, index, _, _ = make_training_data(df, features_and_labels, 0, int)

    # predict probabilities
    y_hat = model.predict(x)
    return x, y, y_hat, index


def _predict(df: pd.DataFrame, model: Model, tail: int = None) -> pd.DataFrame:
    features_and_labels = model.features_and_labels

    if tail is not None:
        if tail <= 0:
            raise ValueError("tail must be > 0 or None")
        elif model.min_required_data is not None:
            # just use the tail for feature engineering
            df = df[-(tail + (model.min_required_data - 1)):]
        else:
            log.warning("could not determine the minimum required data from the model")

    # then re assign data frame with features only
    dff, x = make_forecast_data(df, features_and_labels)

    # first save target columns and loss column
    if features_and_labels.target_columns is not None:
        dff = dff.join(df[features_and_labels.target_columns].add_prefix("traget_"))

    if features_and_labels.loss_column is not None:
        dff["loss"] = df[features_and_labels.loss_column]

    # predict on features
    prediction = model.predict(x)
    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
        for i in range(prediction.shape[1]):
            dff[f"prediction_{model.features_and_labels.labels[i]}"] = prediction[:,i]
    else:
        dff["prediction"] = prediction

    return dff

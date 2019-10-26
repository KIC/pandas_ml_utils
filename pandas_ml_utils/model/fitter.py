import io
import logging
from time import perf_counter
from typing import Callable, Tuple, Dict, Any
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


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
        cross_validation: Tuple[int, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
        cache_feature_matrix: bool = False,
        test_validate_split_seed = 42,
        hyper_parameter_space: Dict = None,
        ) -> Tuple[Model, Tuple, Tuple, Tuple, Any]:
    # get a new model
    trails = None
    model = model_provider()
    features_and_labels = model.features_and_labels

    # make training and test data sets
    x_train, x_test, y_train, y_test, index_train, index_test, min_required_data = \
        make_training_data(df,
                           features_and_labels,
                           test_size,
                           label_type=features_and_labels.label_type,
                           seed=test_validate_split_seed,
                           cache=cache_feature_matrix)

    log.info(f"create model (min required data = {min_required_data}")
    model.min_required_data = min_required_data

    # fit the model
    start_performance_count = log_with_time(lambda: log.info("fit model"))
    if hyper_parameter_space is not None:
        # next isolate hyperopt parameters and constants only used for hyper parameter tuning like early stopping
        constants = {}
        hyperopt_params = {}
        for k, v in list(hyper_parameter_space.items()):
            if k.startswith("__"):
                hyperopt_params[k[2:]] = hyper_parameter_space.pop(k)
            elif type(v) in [int, float, bool]:
                constants[k] = hyper_parameter_space.pop(k)

        # optimize hyper parameters
        model, trails = __hyper_opt(hyper_parameter_space,
                                    hyperopt_params,
                                    constants,
                                    model_provider,
                                    cross_validation,
                                    x_train, y_train, index_train,
                                    x_test, y_test, index_test)

    # finally train the model with eventually tuned hyper parameters
    __train_loop(model, cross_validation, x_train, y_train, index_train, x_test, y_test, index_test)

    log.info(f"fitting model done in {perf_counter() - start_performance_count: .2f} sec!")
    return model, (x_train, y_train), (x_test, y_test), (index_train, index_test), trails


def __train_loop(model, cross_validation, x_train, y_train, index_train,  x_test, y_test, index_test):
    if cross_validation is not None and isinstance(cross_validation, Tuple) and callable(cross_validation[1]):
        losses = []
        for fold_epoch in range(cross_validation[0]):
            # cross validation, make sure we re-shuffle every fold_epoch
            for f, (train_idx, test_idx) in enumerate(cross_validation[1](x_train, y_train)):
                log.info(f'fit fold {f}')
                loss = model.fit(x_train[train_idx], y_train[train_idx], x_train[test_idx], y_train[test_idx],
                                 index_train[train_idx], index_train[test_idx])

                losses.append(loss)

        return np.array(losses).mean()
    else:
        # fit without cross validation
        return model.fit(x_train, y_train, x_test, y_test, index_train, index_test)


@ignore_warnings(category=ConvergenceWarning)
def __hyper_opt(hyper_parameter_space,
                hyperopt_params,
                constants,
                model_provider,
                cross_validation,
                x_train, y_train, index_train,
                x_test, y_test, index_test):
    from hyperopt import fmin, tpe, Trials

    keys = list(hyper_parameter_space.keys())

    def f(args):
        sampled_parameters = {k: args[i] for i, k in enumerate(keys)}
        model = model_provider(**sampled_parameters, **constants)
        loss = __train_loop(model, cross_validation, x_train, y_train, index_train, x_test, y_test, index_test)
        return {'status': 'ok', 'loss': loss, 'parameter': sampled_parameters}

    trails = Trials()
    fmin(f, list(hyper_parameter_space.values()), algo=tpe.suggest, trials=trails, show_progressbar=False, **hyperopt_params)

    # find the best parameters amd make sure to NOT pass the constants as they are only used for hyperopt
    best_parameters = trails.best_trial['result']['parameter']
    best_model = model_provider(**best_parameters)

    print(f'best parameters: {repr(best_parameters)}')
    return best_model, trails


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

    prediction = model.predict(x)
    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
        for i in range(prediction.shape[1]):
            dff[f"prediction_{model.features_and_labels.labels[i]}"] = prediction[:,i]
    else:
        dff["prediction"] = prediction

    return dff

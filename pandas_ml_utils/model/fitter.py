from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable, Tuple, Dict, TYPE_CHECKING
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import numpy as np
import pandas as pd

from ..train_test_data import make_training_data, make_forecast_data
from ..utils import log_with_time
from ..model.models import Model
from ..constants import *

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from hyperopt import Trials


def _fit(df: pd.DataFrame,
         model_provider: Callable[[int], Model],
         test_size: float = 0.4,
         cross_validation: Tuple[int, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
         cache_feature_matrix: bool = False,
         test_validate_split_seed = 42,
         hyper_parameter_space: Dict = None,
         ) -> Tuple[Model, Tuple[pd.DataFrame, pd.DataFrame], Trials]:
    # get a new model
    trails = None
    model = model_provider()
    features_and_labels = model.features_and_labels
    goals = features_and_labels.get_goals()

    # make training and test data sets
    x_train, x_test, y_train, y_test, index_train, index_test = \
        make_training_data(df,
                           features_and_labels,
                           test_size,
                           label_type=features_and_labels.label_type,
                           seed=test_validate_split_seed,
                           cache=cache_feature_matrix)

    _log.info(f"create model (min required data = {features_and_labels.min_required_samples}")

    # eventually perform a hyper parameter optimization first
    start_performance_count = log_with_time(lambda: _log.info("fit model"))
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

    _log.info(f"fitting model done in {perf_counter() - start_performance_count: .2f} sec!")
    header = __stack_header_prediction(goals) + __stack_header_label(goals) + __stack_header_loss(goals)

    df_train = df.loc[index_train]
    df_prediction_train = __predict(df_train, model, x_train) \
        .join(__truth(df_train, model)) \
        .join(__loss(df_train, model))
    df_prediction_train.columns = pd.MultiIndex.from_tuples(header)

    df_prediction_test = None
    if x_test is not None:
        df_test = df.loc[index_test]
        df_prediction_test = __predict(df_test, model, x_test) \
            .join(__truth(df_test, model)) \
            .join(__loss(df_test, model))
        df_prediction_test.columns = pd.MultiIndex.from_tuples(header)

    return model, (df_prediction_train, df_prediction_test), trails


def __train_loop(model, cross_validation, x_train, y_train, index_train,  x_test, y_test, index_test):
    if cross_validation is not None and isinstance(cross_validation, Tuple) and callable(cross_validation[1]):
        losses = []
        for fold_epoch in range(cross_validation[0]):
            # cross validation, make sure we re-shuffle every fold_epoch
            for f, (train_idx, test_idx) in enumerate(cross_validation[1](x_train, y_train)):
                _log.info(f'fit fold {f}')
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
        if loss is None:
            raise ValueError("Can not hyper tune if model loss is None")

        return {'status': 'ok', 'loss': loss, 'parameter': sampled_parameters}

    trails = Trials()
    fmin(f, list(hyper_parameter_space.values()), algo=tpe.suggest, trials=trails, show_progressbar=False, **hyperopt_params)

    # find the best parameters amd make sure to NOT pass the constants as they are only used for hyperopt
    best_parameters = trails.best_trial['result']['parameter']
    best_model = model_provider(**best_parameters)

    print(f'best parameters: {repr(best_parameters)}')
    return best_model, trails


def _backtest(df: pd.DataFrame, model: Model) -> pd.DataFrame:
    features_and_labels = model.features_and_labels
    goals = features_and_labels.get_goals()

    # make training and test data with no 0 test data fraction
    x, _, y, _, index, _ = make_training_data(df, features_and_labels, 0, int)

    # predict probabilities
    df_source = df.loc[index]
    df_backtest = __predict(df_source, model, x) \
        .join(__truth(df_source, model)) \
        .join(__loss(df_source, model))

    header = __stack_header_prediction(goals) + __stack_header_label(goals) + __stack_header_loss(goals)
    df_backtest.columns = pd.MultiIndex.from_tuples(header)

    return df_backtest


def _predict(df: pd.DataFrame, model: Model, tail: int = None) -> pd.DataFrame:
    features_and_labels = model.features_and_labels
    goals = features_and_labels.get_goals()

    if tail is not None:
        if tail <= 0:
            raise ValueError("tail must be > 0 or None")
        elif features_and_labels.min_required_samples is not None:
            # just use the tail for feature engineering
            df = df[-(tail + (features_and_labels.min_required_samples - 1)):]
        else:
            _log.warning("could not determine the minimum required data from the model")

    # predict and return data frame
    dff, x = make_forecast_data(df, features_and_labels)
    df_prediction = __predict(df.loc[dff.index], model, x)

    header = __stack_header_prediction(goals)
    df_prediction.columns = pd.MultiIndex.from_tuples(header)

    return df_prediction


def __predict(df, model, x):
    # first save target columns and loss column
    goals = model.features_and_labels.get_goals()
    predictions = model.predict(x)
    df_pred = pd.DataFrame({}, index=df.index)

    for target, (_, labels) in goals.items():
        prediction = predictions[target]
        postfix = "" if target is None else f'_{target}'

        if target is not None:
            df_pred[f'{TARGET_COLUMN_NAME}_{target}'] = df[target]
        else:
            df_pred[f"{TARGET_COLUMN_NAME}"] = ""

        if len(labels) > 1:
            for i, label in enumerate(labels):
                df_pred[f"{PREDICTION_COLUMN_NAME}{postfix}_{label}"] = prediction[:, i]
        else:
            df_pred[f"{PREDICTION_COLUMN_NAME}{postfix}"] = prediction

    return df_pred


def __loss(df, model):
    df_loss = pd.DataFrame({}, index=df.index)
    goals = model.features_and_labels.get_goals()
    for target, (loss, _) in goals.items():
        postfix = f"_{target}" if len(goals) > 1 else ""
        if loss in df.columns:
            df_loss[f"{LOSS_COLUMN_NAME}{postfix}_{loss}"] = df[loss]
        else:
            df_loss[f"{LOSS_COLUMN_NAME}{postfix}"] = -abs(loss) if loss is not None else -1.0

    return df_loss


def __truth(df, model):
    df_truth = pd.DataFrame({}, index=df.index)
    goals = model.features_and_labels.get_goals()

    for target, (_, labels) in goals.items():
        postfix = "" if target is None else f'_{target}'
        for label in labels:
            postfix2 = "" if len(labels) <= 1 else f"_{label}"
            df_truth[f"{LABEL_COLUMN_NAME}{postfix}{postfix2}"] = df[label]

    return df_truth


def __stack_header_prediction(goals):
    prediction_headers = []
    # prediction
    for target, (loss, labels) in goals.items():
        prediction_headers.append((target or TARGET_COLUMN_NAME, TARGET_COLUMN_NAME, "value"))
        for l in labels:
            prediction_headers.append((target or TARGET_COLUMN_NAME, PREDICTION_COLUMN_NAME, l if len(labels) > 1 else "value"))

    return prediction_headers


def __stack_header_label(goals):
    return [(target or TARGET_COLUMN_NAME, LABEL_COLUMN_NAME, l if len(labels) > 1 else "value") for target, (_, labels) in goals.items() for l in labels ]


def __stack_header_loss(goals):
    return [(target or TARGET_COLUMN_NAME, LOSS_COLUMN_NAME, "value") for target, (loss, _) in goals.items()]

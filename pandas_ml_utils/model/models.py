from __future__ import annotations

import inspect
import logging
from copy import deepcopy

import dill as pickle
import numpy as np
import pandas as pd
from typing import List, Callable, Tuple, TYPE_CHECKING, Dict

from sklearn.linear_model import LogisticRegression

from pandas_ml_utils.summary.summary import Summary
from pandas_ml_utils.model.features_and_labels.features_and_labels import FeaturesAndLabels

_log = logging.getLogger(__name__)


class Model(object):
    """
    Represents a statistical or ML model and holds the necessary information how to interpret the columns of a
    pandas *DataFrame* ( :class:`.FeaturesAndLabels` ). Currently available implementations are:
     * SkitModel - provide any skit learn classifier or regressor
     * KerasModel - provide a function returning a compiled keras model
     * MultiModel - provide a model which will copied (and fitted) for each provided target
    """

    @staticmethod
    def load(filename: str):
        """
        Loads a previously saved model from disk. By default `dill <https://pypi.org/project/dill/>`_ is used to
        serialize / deserialize a model.

        :param filename: filename of the serialized model
        :return: returns a deserialized model
        """
        with open(filename, 'rb') as file:
            model = pickle.load(file)
            if isinstance(model, Model):
                return model
            else:
                raise ValueError("Deserialized pickle was not a Model!")

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[pd.DataFrame], Summary] = Summary,
                 **kwargs):
        """
        All implementations of `Model` need to pass two arguments to `super().__init()__`.

        :param features_and_labels: the :class:`.FeaturesAndLabels` object defining all the features,
                                    feature engineerings and labels
        :param summary_provider: a summary provider in the most simple case just holds a `pd.DataFrame` containing all
                                 the labels and all the predictions and optionally loss and target values. Since
                                 constructors as callables as well it is usually enoug tho just pass the type i.e.
                                 `summary_provider=BinaryClassificationSummary`
        :param kwargs:
        """
        self._features_and_labels = features_and_labels
        self._summary_provider = summary_provider
        self.kwargs = kwargs

    @property
    def features_and_labels(self):
        return self._features_and_labels

    @property
    def summary_provider(self):
        return self._summary_provider

    def plot_loss(self):
        pass

    def __getitem__(self, item):
        """
        returns arguments which are stored in the kwargs filed. By providing a tuple, a default in case of missing
        key can be specified
        :param item: name of the item im the kwargs dict or tuple of name, default
        :return: item or default
        """
        if isinstance(item, tuple) and len(item) == 2:
            return self.kwargs[item[0]] if item[0] in self.kwargs else item[1]
        else:
            return self.kwargs[item] if item in self.kwargs else None

    def save(self, filename: str):
        """
        save model to disk
        :param filename: filename
        :return: None
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def fit(self, x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, df_index_train: list, df_index_test: list) -> float:
        """
        function called to fit the model
        :param x: x
        :param y: y
        :param x_val: x validation
        :param y_val: y validation
        :param df_index_train: index of x, y values in the DataFrame
        :param df_index_test: index of x_val, y_val values in the DataFrame
        :return: loss of the fit
        """
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        prediction of the model for each target

        :param x: x
        :return: prediction of the model for each target
        """

        pass

    def __call__(self, *args, **kwargs):
        """
        returns a copy pf the model with eventually different configuration (kwargs). This is useful for hyper paramter
        tuning or for MultiModels

        :param args:
        :param kwargs: arguments which are eventually provided by hyperopt or by different targets
        :return:
        """
        if not kwargs:
            return deepcopy(self)
        else:
            raise ValueError(f"construction of model with new parameters is not supported\n{type(self)}: {kwargs}")


class SkitModel(Model):

    def __init__(self,
                 skit_model,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[pd.DataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.skit_model = skit_model

    def fit(self, x, y, x_val, y_val, df_index_train, df_index_test):
        # shape correction if needed
        y = y.ravel() if len(y.shape) > 1 and y.shape[1] == 1 else y

        # remember fitted model
        self.skit_model = self.skit_model.fit(SkitModel.reshape_rnn_as_ar(x), y)

        if getattr(self.skit_model, 'loss_', None):
            return self.skit_model.loss_
        else:
            prediction = self.predict(x)
            if type(self.skit_model) == LogisticRegression\
            or type(self.skit_model).__name__.endswith("Classifier")\
            or type(self.skit_model).__name__.endswith("SVC"):
                from sklearn.metrics import log_loss
                try:
                    return log_loss(prediction > 0.5, y).mean()
                except ValueError as e:
                    if "contains only one label" in str(e):
                        return -100
                    else:
                        raise e
            else:
                from sklearn.metrics import mean_squared_error
                return mean_squared_error(prediction, y).mean()

    def predict(self, x) -> np.ndarray:
        if callable(getattr(self.skit_model, 'predict_proba', None)):
            y_hat = self.skit_model.predict_proba(SkitModel.reshape_rnn_as_ar(x))
            return y_hat[:, 1] if len(self.features_and_labels.labels) == 1 and y_hat.shape[1] == 2 else y_hat
        else:
            return self.skit_model.predict(SkitModel.reshape_rnn_as_ar(x))

    def plot_loss(self):
        loss_curve = getattr(self.skit_model, 'loss_curve_', None)

        if loss_curve is not None:
            import matplotlib.pyplot as plt
            plt.plot(loss_curve)
        else:
            print("no loss curve found")

    def __str__(self):
        return f'{__name__}({repr(self.skit_model)}, {self.features_and_labels})'

    def __call__(self, *args, **kwargs):
        if not kwargs:
            return deepcopy(self)
        else:
            new_model = SkitModel(type(self.skit_model)(**kwargs), self.features_and_labels, self.summary_provider)
            new_model.kwargs = deepcopy(self.kwargs)
            return new_model

    @staticmethod
    def reshape_rnn_as_ar(arr3d):
        if len(arr3d.shape) < 3:
            print("Data was not in RNN shape")
            return arr3d
        else:
            return arr3d.reshape(arr3d.shape[0], arr3d.shape[1] * arr3d.shape[2])


class KerasModel(Model):
    # eventually we need to save and load the weights of the keras model individually by using `__getstate__`
    #  `__setstate__` like described here: http://zachmoshe.com/2017/04/03/pickling-keras-models.html
    if TYPE_CHECKING:
        from keras.models import Model as KModel

    def __init__(self,
                 keras_compiled_model_provider: Callable[[], KModel],
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[pd.DataFrame], Summary] = Summary,
                 epochs: int = 100,
                 callbacks: List[Callable] = [],
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.keras_model_provider = keras_compiled_model_provider
        self.keras_model = keras_compiled_model_provider()
        self.epochs = epochs
        self.callbacks = callbacks
        self.history = None

    def fit(self, x, y, x_val, y_val, df_index_train, df_index_test) -> float:
        possible_fitter_args = inspect.getfullargspec(self.keras_model.fit).args
        fitter_args = {arg: self.kwargs[arg] for arg in self.kwargs.keys() if arg in possible_fitter_args}
        fit_history = self.keras_model.fit(x, y, epochs=self.epochs, validation_data=(x_val, y_val), callbacks=self.callbacks, **fitter_args)

        if self.history is None:
            self.history = fit_history.history
        else:
            for metric, values in self.history.items():
                self.history[metric] = self.history[metric] + fit_history.history[metric]

        return min(fit_history.history['loss'])

    def predict(self, x):
        return self.keras_model.predict(x)

    def plot_loss(self):
        import matplotlib.pyplot as plt

        plt.plot(self.history['val_loss'])
        plt.plot(self.history['loss'])

    def __call__(self, *args, **kwargs):
        new_model = KerasModel(self.keras_model_provider,
                               self.features_and_labels,
                               self.summary_provider,
                               self.epochs,
                               deepcopy(self.callbacks),
                               **deepcopy(self.kwargs))

        if kwargs:
            new_model.keras_model = new_model.keras_model_provider(**kwargs)

        return new_model


class MultiModel(Model):

    def __init__(self,
                 model_provider: Model,
                 summary_provider: Callable[[pd.DataFrame], Summary] = Summary,
                 alpha: float = 0.5):
        super().__init__(model_provider.features_and_labels, summary_provider)

        if isinstance(model_provider, MultiModel):
            raise ValueError("Nesting Multi Models is not supported, you might use a flat structure of all your models")

        self.model_provider = model_provider
        self.models = {target: model_provider() for target in self.features_and_labels.labels.keys()}
        self.alpha = alpha

    def fit(self, x, y, x_val, y_val, df_index_train, df_index_test) -> float:
        losses = []
        pos = 0
        for target, labels in self.features_and_labels.labels.items():
            index = range(pos, pos + len(labels))
            target_y = y[:,index]
            target_y_val = y_val[:,index]
            _log.info(f"fit model for target {target}")
            losses.append(self.models[target].fit(x, target_y, x_val, target_y_val, df_index_train, df_index_test))
            pos += len(labels)

        losses = np.array(losses)
        a = self.alpha

        # return weighted loss between mean and max loss
        return (losses.mean() * (1 - a) + a * losses.max()) if len(losses) > 0 else None

    def predict(self, x: np.ndarray) -> np.ndarray:
        # intermediate function to concatenate individual predictions
        def predict_as_column_matrix(target):
            prediction = self.models[target].predict(x)

            # eventually fix shape to have 2 dimensions
            if len(prediction.shape) <= 1:
                prediction = prediction.reshape((-1, 1))

            # fix dimensions if prediction length is 2 and expected length is 1
            if prediction.shape[1] == 2 and len(self.features_and_labels.labels[target]) == 1:
                prediction = prediction[:, 1].reshape((-1, 1))

            # return prediction with expected shape
            return prediction

        # return all the concatenated predictions
        return np.concatenate([predict_as_column_matrix(target) for target in self.features_and_labels.labels.keys()],
                              axis=1)

    def __call__(self, *args, **kwargs):
        new_multi_model = MultiModel(self.model_provider, self.summary_provider, self.alpha)

        if kwargs:
            new_multi_model.models = {target: self.model_provider(**kwargs) for target in self.features_and_labels.get_goals().keys()}

        return new_multi_model



## THIS need to be fixed somewhen
#class OpenAiGymModel(Model):
#    from typing import TYPE_CHECKING
#    if TYPE_CHECKING:
#        from rl.core import Agent
#
#    def __init__(self,
#                 agent_provider: Callable[[Tuple, int], Agent],
#                 features_and_labels: FeaturesAndLabels,
#                 action_reward_functions: List[Callable[[np.ndarray], float]],
#                 reward_range: Tuple[int, int],
#                 oservation_range: Tuple[int, int] = None,
#                 episodes: int = 1000,
#                 **kwargs):
#        super().__init__(features_and_labels, **kwargs)
#        self.agent_provider = agent_provider
#        self.action_reward_functions = action_reward_functions
#        self.reward_range = reward_range
#        self.oservation_range = oservation_range
#        self.episodes = episodes
#        self.agent = agent_provider(features_and_labels.shape()[0], len(action_reward_functions))
#
#        # some history
#        self.keras_train_history = None
#        self.keras_test_history = None
#        self.history = ()
#
#    def fit(self, x, y, x_val, y_val, df_index_train, df_index_test):
#        mm = (min([x.min(), x_val.min()]), max([x.max(), x_val.max()])) if self.oservation_range is None else self.oservation_range
#        training_gym = RowWiseGym((df_index_train, x, y), self.features_and_labels, self.action_reward_functions, self.reward_range, mm)
#        test_gym = RowWiseGym((df_index_test, x_val, y_val), self.features_and_labels, self.action_reward_functions, self.reward_range, mm)
#
#        self.keras_train_history = self.agent.fit(training_gym, nb_steps=len(x) * self.episodes)
#        #self.keras_test_history = self.agent.test(test_gym, nb_episodes=1) # clarification needed what test actually does: https://github.com/keras-rl/keras-rl/issues/342
#        test_gym = self._forward_gym(test_gym)
#
#        self.history = (training_gym.get_history(), test_gym.get_history())
#
#    def back_test(self, index, x, y):
#        mm = (x.min(), x.max()) if self.oservation_range is None else self.oservation_range
#        gym = RowWiseGym((index, x, y), self.features_and_labels, self.action_reward_functions, self.reward_range, mm)
#        return self._forward_gym(gym).get_history()
#
#    def _predict(self, x: np.ndarray, target: str) -> np.ndarray:
#        return [self.agent.forward(x[r]) for r in range(len(x))]
#
#    def __call__(self, *args, **kwargs):
#        if kwargs:
#            raise ValueError("hyper parameter tunig currently not supported for RL")
#
#        return OpenAiGymModel(self.agent_provider,
#                              self.features_and_labels,
#                              self.action_reward_functions,
#                              self.reward_range,
#                              self.episodes,
#                              **deepcopy(self.kwargs))
#
#    def _forward_gym(self, gym):
#        done = False
#        state = gym.reset()
#        while not done:
#            state, reward, done, _ = gym.step(self.agent.forward(state))
#
#        return gym
#
#class StableBaselineModel(Model):
#    # add stable baseline models https://stable-baselines.readthedocs.io/en/master/guide/algos.html
#    pass
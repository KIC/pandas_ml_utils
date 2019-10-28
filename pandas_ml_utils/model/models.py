from __future__ import annotations
import logging
from copy import deepcopy

import dill as pickle
import numpy as np
from typing import List, Callable, Tuple, TYPE_CHECKING

from sklearn.linear_model import LogisticRegression

from .features_and_Labels import FeaturesAndLabels
from ..train_test_data import reshape_rnn_as_ar
from ..reinforcement.gym import RowWiseGym
from ..extern.loss_functions import mse

log = logging.getLogger(__name__)


class Model(object):

    @staticmethod
    def load(filename: str):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
            if isinstance(model, Model):
                return model
            else:
                raise ValueError("Deserialized pickle was not a Model!")

    def __init__(self, features_and_labels: FeaturesAndLabels, **kwargs):
        self.features_and_labels = features_and_labels
        self.min_required_data: int = None
        self.kwargs = kwargs

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == 2:
            return self.kwargs[item[0]] if item[0] in self.kwargs else item[1]
        else:
            return self.kwargs[item] if item in self.kwargs else None

    def save(self, filename: str):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def fit(self, x, y, x_val, y_val, df_index_train, df_index_test) -> float:
        pass

    def predict(self, x) -> np.ndarray:
        pass

    # this lets the model itself act as a provider. However we want to use the same Model configuration
    # for different datasets (i.e. as part of MultiModel)
    def __call__(self, *args, **kwargs):
        if not kwargs:
            return deepcopy(self)
        else:
            raise ValueError(f"construction of model with new parameters is not supported\n{type(self)}: {kwargs}")


class SkitModel(Model):

    def __init__(self, skit_model, features_and_labels: FeaturesAndLabels, **kwargs):
        super().__init__(features_and_labels, **kwargs)
        self.skit_model = skit_model

    def fit(self, x, y, x_val, y_val, df_index_train, df_index_test):
        # remember fitted model
        self.skit_model = self.skit_model.fit(reshape_rnn_as_ar(x), y)

        if getattr(self.skit_model, 'loss_', None):
            return self.skit_model.loss_
        else:
            if type(self.skit_model) == LogisticRegression\
            or type(self.skit_model).__name__.endswith("Classifier")\
            or type(self.skit_model).__name__.endswith("SVC"):
                from sklearn.metrics import log_loss
                return log_loss(self.predict(x) > 0.5, y)
            else:
                from sklearn.metrics import mean_squared_error
                return mean_squared_error(self.predict(x), y)

    def predict(self, x):
        if callable(getattr(self.skit_model, 'predict_proba', None)):
            return self.skit_model.predict_proba(reshape_rnn_as_ar(x))[:, 1]
        else:
            return self.skit_model.predict(reshape_rnn_as_ar(x))

    def __str__(self):
        return f'{__name__}({repr(self.skit_model)}, {self.features_and_labels})'

    def __call__(self, *args, **kwargs):
        if not kwargs:
            return deepcopy(self)
        else:
            new_model = SkitModel(type(self.skit_model)(**kwargs), self.features_and_labels)
            new_model.kwargs = deepcopy(self.kwargs)
            return new_model


class KerasModel(Model):
    # eventually we need to save and load the weights of the keras model individually by using `__getstate__`
    #  `__setstate__` like described here: http://zachmoshe.com/2017/04/03/pickling-keras-models.html
    if TYPE_CHECKING:
        from keras.models import Model as KModel

    def __init__(self,
                 keras_compiled_model_provider: Callable[[], KModel],
                 features_and_labels: FeaturesAndLabels,
                 callbacks: List[Callable],
                 **kwargs):
        super().__init__(features_and_labels, **kwargs)
        self.keras_model_provider = keras_compiled_model_provider
        self.keras_model = keras_compiled_model_provider()
        self.callbacks = callbacks
        self.history = None

    def fit(self, x, y, x_val, y_val, df_index_train, df_index_test) -> float:
        self.history = self.keras_model.fit(x, y, validation_data=(x_val, y_val), callbacks=self.callbacks)
        return min(self.history.history['loss'])

    def predict(self, x) -> np.ndarray:
        self.keras_model.predict(x)

    def __call__(self, *args, **kwargs):
        new_model = KerasModel(self.keras_model_provider,
                               self.features_and_labels,
                               deepcopy(self.callbacks),
                               **deepcopy(self.kwargs))

        if kwargs:
            new_model.keras_model = new_model.keras_model_provider(**kwargs)

        return new_model


class MultiModel(Model):

    def __init__(self, model_provider: Model, alpha: float = 0.5):
        super().__init__(model_provider.features_and_labels)
        self.model_provider = model_provider
        self.models = {target: model_provider() for target in self.features_and_labels.get_goals().keys()}
        self.alpha = alpha

    def fit(self, x, y, x_val, y_val, df_index_train, df_index_test) -> float:
        goals = self.features_and_labels.get_goals()
        losses = []
        for target, (_, labels) in goals.items():
            index = [self.features_and_labels.labels.index(label) for label in labels]
            target_y = y[:,index]
            target_y_val = y_val[:,index]
            log.info(f"fit model for target {target}")
            losses.append(self.models[target].fit(x, target_y, x_val, target_y_val, df_index_train, df_index_test))

        losses = np.array(losses)
        a = self.alpha

        # return weighted loss between mean and max loss
        return (losses.mean() * (1 - a) + a * losses.max()) if len(losses) > 0 else None

    def predict(self, x) -> np.ndarray:
        predictions = {target: self.models[target].predict(x) for target, _ in self.models.items()}
        # TODO think about how to return a multi model prediction ...
        pass

    def __call__(self, *args, **kwargs):
        new_multi_model = MultiModel(self.model_provider, self.alpha)
        if kwargs:
            new_multi_model.models = {target: self.model_provider(**kwargs) for target in self.features_and_labels.get_goals().keys()}

        return new_multi_model


class OpenAiGymModel(Model):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from rl.core import Agent

    def __init__(self,
                 agent_provider: Callable[[Tuple, int], Agent],
                 features_and_labels: FeaturesAndLabels,
                 action_reward_functions: List[Callable[[np.ndarray], float]],
                 reward_range: Tuple[int, int],
                 oservation_range: Tuple[int, int] = None,
                 episodes: int = 1000,
                 **kwargs):
        super().__init__(features_and_labels, **kwargs)
        self.agent_provider = agent_provider
        self.action_reward_functions = action_reward_functions
        self.reward_range = reward_range
        self.oservation_range = oservation_range
        self.episodes = episodes
        self.agent = agent_provider(features_and_labels.shape()[0], len(action_reward_functions))

        # some history
        self.keras_train_history = None
        self.keras_test_history = None
        self.history = ()

    def fit(self, x, y, x_val, y_val, df_index_train, df_index_test):
        mm = (min([x.min(), x_val.min()]), max([x.max(), x_val.max()])) if self.oservation_range is None else self.oservation_range
        training_gym = RowWiseGym((df_index_train, x, y), self.features_and_labels, self.action_reward_functions, self.reward_range, mm)
        test_gym = RowWiseGym((df_index_test, x_val, y_val), self.features_and_labels, self.action_reward_functions, self.reward_range, mm)

        self.keras_train_history = self.agent.fit(training_gym, nb_steps=len(x) * self.episodes)
        #self.keras_test_history = self.agent.test(test_gym, nb_episodes=1) # clarification needed what test actually does: https://github.com/keras-rl/keras-rl/issues/342
        test_gym = self._forward_gym(test_gym)

        self.history = (training_gym.get_history(), test_gym.get_history())

    def back_test(self, index, x, y):
        mm = (x.min(), x.max()) if self.oservation_range is None else self.oservation_range
        gym = RowWiseGym((index, x, y), self.features_and_labels, self.action_reward_functions, self.reward_range, mm)
        return self._forward_gym(gym).get_history()

    def predict(self, x):
        return [self.agent.forward(x[r]) for r in range(len(x))]

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise ValueError("hyper parameter tunig currently not supported for RL")

        return OpenAiGymModel(self.agent_provider,
                              self.features_and_labels,
                              self.action_reward_functions,
                              self.reward_range,
                              self.episodes,
                              **deepcopy(self.kwargs))

    def _forward_gym(self, gym):
        done = False
        state = gym.reset()
        while not done:
            state, reward, done, _ = gym.step(self.agent.forward(state))

        return gym

class StableBaselineModel(Model):
    # add stable baseline models https://stable-baselines.readthedocs.io/en/master/guide/algos.html
    pass
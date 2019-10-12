from __future__ import annotations
import logging
from copy import deepcopy

import dill as pickle
import numpy as np
from typing import List, Callable, Tuple

from .features_and_Labels import FeaturesAndLabels
from ..train_test_data import reshape_rnn_as_ar
from ..reinforcement.gym import RowWiseGym

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

    def fit(self, x, y, x_val, y_val, df_index_train, df_index_test) -> None:
        pass

    def predict(self, x) -> np.ndarray:
        pass

    # this lets the model itself act as a provider. However we want to use the same Model configuration
    # for different datasets (i.e. as part of MultiModel)
    def __call__(self, *args, **kwargs):
        return deepcopy(self)


class SkitModel(Model):

    def __init__(self, skit_model, features_and_labels: FeaturesAndLabels, **kwargs):
        super().__init__(features_and_labels, **kwargs)
        self.skit_model = skit_model

    def fit(self, x, y, x_val, y_val, df_index_train, df_index_test):
        self.skit_model.fit(reshape_rnn_as_ar(x), y),

    def predict(self, x):
        if callable(getattr(self.skit_model, 'predict_proba', None)):
            return self.skit_model.predict_proba(reshape_rnn_as_ar(x))[:, 1]
        else:
            return self.skit_model.predict(reshape_rnn_as_ar(x))


# TODO add Keras Model
class KerasModel(Model):

    def __init__(self, keras_model_provider, features_and_labels: FeaturesAndLabels, **kwargs):
        super().__init__(features_and_labels, **kwargs)
        self.keras_model_provider = keras_model_provider

    def fit(self, x, y, x_val, y_val, df_index_train, df_index_test) -> None:
        pass

    def predict(self, x) -> np.ndarray:
        # we would need to return a prediction for every and each parameters dict in the parameter space
        pass

    def __call__(self, *args, **kwargs):
        return KerasModel(self.keras_model_provider,
                          self.features_and_labels,
                          **self.kwargs)


# class MultiModel(Model):
#
#     def __init__(self, model_provider: Callable[[], Model], features_and_labels: FeaturesAndLabels):
#         super().__init__(features_and_labels)
#         self.model_provider = model_provider
#
#     def fit(self, x, y, x_val, y_val) -> None:
#         pass
#
#     def predict(self, x) -> np.ndarray:
#         # we would need to return a prediction for every and each parameters dict in the parameter space
#         pass


class OpenAiGymModel(Model):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from rl.core import Agent

    def __init__(self,
                 agent_provider: Callable[[Tuple, int], Agent],
                 features_and_labels: FeaturesAndLabels,
                 action_reward_functions: List[Callable[[np.ndarray], float]],
                 reward_range: Tuple[int, int],
                 episodes: int = 1000,
                 **kwargs):
        super().__init__(features_and_labels, **kwargs)
        self.agent_provider = agent_provider
        self.action_reward_functions = action_reward_functions
        self.reward_range = reward_range
        self.episodes = episodes
        self.agent = agent_provider(features_and_labels.shape()[0], len(action_reward_functions))

    def fit(self, x, y, x_val, y_val, df_index_train, df_index_test):
        training_gym = RowWiseGym((df_index_train, x, y), self.features_and_labels, self.action_reward_functions, self.reward_range)
        test_gmy = RowWiseGym((df_index_test, x_val, y_val), self.features_and_labels, self.action_reward_functions, self.reward_range)

        keras_train_history = self.agent.fit(training_gym, nb_steps=len(x) * self.episodes)
        keras_test_history =  self.agent.test(test_gmy, nb_episodes=1)

        training_gym.get_history()
        return None # FIXME return Fit with som information of the total reward

    def predict(self, x):
        # gym = RowWiseGym(self.features_and_labels, self.action_reward_functions, self.reward_range)
        # fixme
        pass

    def __call__(self, *args, **kwargs):
        return OpenAiGymModel(self.agent_provider,
                              self.features_and_labels,
                              self.action_reward_functions,
                              self.reward_range,
                              self.episodes,
                              **self.kwargs)

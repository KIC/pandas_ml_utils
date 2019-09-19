import logging

import dill as pickle
import numpy as np

from copy import deepcopy

from typing import Callable

from .train_test_data import reshape_rnn_as_ar
from .features_and_Labels import FeaturesAndLabels

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

    def __init__(self, features_and_labels: FeaturesAndLabels):
        self.features_and_labels = features_and_labels
        self.min_required_data: int = None

    def save(self, filename: str):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def fit(self, x, y, x_val, y_val) -> None:
        pass

    def predict(self, x) -> np.ndarray:
        pass

    # this lets the model also act as a provider
    def __call__(self, *args, **kwargs):
        return deepcopy(self)


class SkitModel(Model):

    def __init__(self, skit_model, features_and_labels: FeaturesAndLabels):
        super().__init__(features_and_labels)
        self.skit_model = skit_model

    def fit(self, x, y, x_val, y_val):
        self.skit_model.fit(reshape_rnn_as_ar(x), y),

    def predict(self, x):
        return self.skit_model.predict_proba(reshape_rnn_as_ar(x))[:, 1]


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

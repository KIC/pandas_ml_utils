import logging
from copy import deepcopy

import dill as pickle
import numpy as np

from .features_and_Labels import FeaturesAndLabels
from ..train_test_data import reshape_rnn_as_ar

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

    def fit(self, x, y, x_val, y_val) -> None:
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

    def fit(self, x, y, x_val, y_val):
        self.skit_model.fit(reshape_rnn_as_ar(x), y),

    def predict(self, x):
        if callable(getattr(self.skit_model, 'predict_proba', None)):
            return self.skit_model.predict_proba(reshape_rnn_as_ar(x))[:, 1]
        else:
            return self.skit_model.predict(reshape_rnn_as_ar(x))


# TODO add Keras Model
class KerasModel(Model):
    pass

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

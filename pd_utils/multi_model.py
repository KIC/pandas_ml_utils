import logging
from collections.abc import Iterable

from typing import Tuple, Dict, Callable, Any, Union, List
from .data_objects import FeaturesAndLabels, Model
import dill as pickle
import pandas as pd
import numpy as np


log = logging.getLogger(__name__)


class MultiModel(object):

    @staticmethod
    def load(filename: str):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def __init__(self,
                 data_provider: Callable[[], pd.DataFrame],
                 data_engineer: Callable[[pd.DataFrame], pd.DataFrame],
                 model_provider: Callable[[], Model],
                 features_and_labels: FeaturesAndLabels,
                 parameter_space: Dict[str, Iterable]):
        self.data_provider = data_provider
        self.data_engineer = data_engineer
        self.model_provider = model_provider
        self.features_and_labels = features_and_labels
        self.parameter_space = self._evaluate_full_parameter_space(parameter_space.copy(), {})
        self.data: pd.DataFrame = None
        self.classification_summaries = None

    def save(self, filename: str):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def _evaluate_full_parameter_space(self, parameter_space: Dict[str, Iterable], parameters: Dict[str, Any]):
        if len(parameter_space) > 0:
            # more parameters need to be unfolded
            parameter, space = parameter_space.popitem()
            return [self._evaluate_full_parameter_space(parameter_space.copy(), {**parameters, parameter: argument}) for argument in space]
        else:
            # all needed parameters are unfolded, execute code using parameters as **kwargs
            return parameters

    # def fetch_data_and_fit(self, test_size: float = 0.4, test_validate_split_seed: int=None):
    #     self.fetch_data()
    #     self.fit()

    def fetch_data(self):
        self.data = self.data_provider()

    def fit(self, test_size: float = 0.4, test_validate_split_seed: int=None):
        def model_fitter(**kwargs):
            self.data_engineer(self.data, **kwargs) \
                .fit_classifier(self.features_and_labels,
                                self.model_provider,
                                test_size=test_size,
                                test_validate_split_seed=test_validate_split_seed)

        self.classification_summaries = self._execute_on_parameter_space(self.parameter_space, model_fitter)

    def predict(self):
        # return a prediction of every model
        pass

    def _execute_on_parameter_space(self, space: Union[List, Dict], script: Callable):
        if isinstance(space, list):
            return [self._execute_on_parameter_space(subspace, script) for subspace in space]
        else:
            log.debug(f"call script for parameters: { {**space} }")
            return script(**space)



from time import perf_counter as pc
from typing import Callable, Dict, Iterable, Any, List

import numpy as np
from sklearn.model_selection import BaseCrossValidator


def log_with_time(log_statement: Callable[[], None]):
    log_statement()
    return pc()


def unfold_parameter_space(parameter_space: Dict[str, Iterable], parameters: Dict[str, Any]) -> List[Dict]:
    if len(parameter_space) > 0:
        # more parameters need to be unfolded
        parameter, space = parameter_space.popitem()
        return list(np.array([unfold_parameter_space(parameter_space.copy(), {**parameters, parameter: argument}) for
                              argument in space]).flat)
    else:
        return parameters


class kfold_with_overweigthing(BaseCrossValidator):
    def _iter_test_indices(self, X=None, y=None, groups=None):
        pass

    def get_n_splits(self, X=None, y=None, groups=None):
        pass
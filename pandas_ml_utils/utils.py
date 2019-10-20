from time import perf_counter as pc
from typing import Callable, Dict, Iterable, Any, List, Tuple

import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils.validation import _num_samples, check_random_state, indexable


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


class KFoldBoostRareEvents(KFold):

    def __init__(self, n_splits='warn', shuffle=False, random_state=None):
        super().__init__(n_splits, shuffle, random_state)

    def split(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        rare_event_indices = indices[y >= 0.999]

        for f, (train_idx, test_idx) in enumerate(super().split(X, y, groups)):
            yield np.hstack([train_idx, rare_event_indices]), np.hstack([test_idx, rare_event_indices])


class ReScaler(object):

    def __init__(self, domain: Tuple[float], range: Tuple[float]):
        self.domain = domain
        self.range = range
        self.resacle = np.vectorize(self._rescale)

    def _interpolate(self, x: float):
        return self.range[0] * (1 - x) + self.range[1] * x

    def _uninterpolate(self, x: float):
        b = (self.domain[1] - self.domain[0]) if (self.domain[1] - self.domain[0]) != 0 else (1 / self.domain[1])
        return (x - self.domain[0]) / b

    def _rescale(self, x: float):
        return self._interpolate(self._uninterpolate(x))

    def __call__(self, *args, **kwargs):
        return self.resacle(args[0])
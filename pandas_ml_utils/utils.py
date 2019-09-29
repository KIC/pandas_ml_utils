from time import perf_counter as pc
from typing import Callable, Dict, Iterable, Any, List

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

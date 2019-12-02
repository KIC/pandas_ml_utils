import io
import base64
from collections import OrderedDict
from time import perf_counter as pc
from typing import Callable, Dict, Iterable, Any, List

import numpy as np


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


def unique(items):
    return list(OrderedDict.fromkeys(items))


def fig_to_png_base64(fig):
    import matplotlib.pyplot as plt
    with io.BytesIO() as f:
        fig.savefig(f, format="png", bbox_inches='tight')
        image = base64.encodebytes(f.getvalue()).decode("utf-8")
        plt.close(fig)
        return image


def one_hot(index: int, number_of_classes: int):
    vec = np.zeros(number_of_classes)

    if index >= 0:
        vec[index] = 1

    return vec


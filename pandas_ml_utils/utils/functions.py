import io
import base64
import inspect
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


def suitable_kwargs(func, **kwargs):
    suitable_args = inspect.getfullargspec(func).args
    return {arg: kwargs[arg] for arg in kwargs.keys() if arg in suitable_args}


def call_with_suitable_kwargs(func, **kwargs):
    args = suitable_kwargs(func, **kwargs)
    return func(**args)


def call_callable_dynamic_args(func, *args, **kwargs):
    spec = inspect.getfullargspec(func)
    call_args = []

    for i in range(len(spec.args)):
        if i < len(args):
            call_args.append(args[i])
        elif spec.args[i] in kwargs:
            call_args.append(kwargs[spec.args[i]])
            del kwargs[spec.args[i]]

    # inject eventually still missing var args
    if spec.varargs and len(args) > len(spec.args) and len(args) > len(call_args):
        call_args += args[len(call_args):]

    # inject kwargs if we have some left overs
    if spec.varkw:
        return func(*call_args, **kwargs)
    else:
        return func(*call_args)

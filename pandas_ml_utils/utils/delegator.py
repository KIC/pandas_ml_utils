from delegateto import DelegateTo
from typing import Tuple, Type
from pandas import DataFrame


def _wrap(func: callable, wrap_class_to: Tuple[Type, Type]):
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        return wrap_class_to[1](res) if type(res) == wrap_class_to[0] else res

    return wrapper


class DelegateToWithWrapper(DelegateTo):

    def __init__(self, wrap_class_to: Tuple[Type, Type], to: str, method=None):
        self.wrap_class_to = wrap_class_to
        super().__init__(to, method)

    def __get__(self, obj, objtype):
        prop = super().__get__(obj, objtype)

        if callable(prop):
            return _wrap(prop, self.wrap_class_to)
        elif type(prop) == self.wrap_class_to[0]:
            return self.wrap_class_to[1](prop)
        else:
            return prop


def delegate_to(to, *classes):
    parent_methods = {}
    for c in classes:
        parent_methods = {*parent_methods, *dir(c)}

    def dec(klass):
        methods = parent_methods - set(dir(klass))
        for m in methods:
            setattr(klass, m, DelegateToWithWrapper((classes[0], klass), to, m))

        return klass

    return dec

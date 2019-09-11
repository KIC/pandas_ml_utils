from time import perf_counter as pc

from typing import Callable


def log_with_time(log_statement: Callable[[], None]):
    log_statement()
    return pc()

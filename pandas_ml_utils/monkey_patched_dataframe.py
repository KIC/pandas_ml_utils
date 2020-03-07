import logging

import numpy as _np
import pandas as _pd
from pandas import *

from pandas_ml_utils.utils.functions import unique_top_level_columns, integrate_nested_arrays

_log = logging.getLogger(__name__)
__version__ = f'patched: {_pd.__version__}'


def _values2D(self):
    values = self.values
    return integrate_nested_arrays(values)


def _values3D(self):
    values = integrate_nested_arrays(self.values)

    if isinstance(self.columns, MultiIndex):
        top_level_columns = unique_top_level_columns(self)

        # features eventually are in [feature, row, time_step]
        # but need to be in RNN shape which is [row, time_step, feature]
        values3D = values if top_level_columns is None else \
            _np.array([self[top_level_col].values for top_level_col in top_level_columns],
                     ndmin=3).swapaxes(0, 1).swapaxes(1, 2)

        if len(values3D) <= 0:
            _log.warning("empty values array!")

        return values3D
    else:
        # a normal data frame with shape [row, time_steps]
        # but need to be in RNN shape which is [row, time_step, feature]
        return values #values.reshape((*values.shape, 1))


DataFrame.label_values = property(_values2D)
DataFrame.feature_values = property(_values3D)

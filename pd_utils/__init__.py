import pandas as pd

from .utils import *
from .lazy_dataframe import *
from .fetch_yahoo import *
from pandas.core.base import PandasObject


# add functions to pandas
PandasObject.inner_join = inner_join
PandasObject.drop_re = drop_re
PandasObject.add_apply = add_apply
PandasObject.shift_inplace = shift_inplace

setattr(pd, 'fetch_yahoo', fetch_yahoo)
from .utils import *
from pandas.core.base import PandasObject


# add functions to pandas
PandasObject.drop_re = drop_re
PandasObject.add_apply = add_apply
PandasObject.shift_inplace = shift_inplace
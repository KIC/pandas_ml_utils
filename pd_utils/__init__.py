import pandas as pd

from .utils import *
from .lazy_dataframe import *
from .fetch_yahoo import *
from .training_test_data import *
from pandas.core.base import PandasObject


# add functions to pandas
PandasObject.inner_join = inner_join
PandasObject.drop_re = drop_re
PandasObject.add_apply = add_apply
PandasObject.shift_inplace = shift_inplace
PandasObject.extend_forecast = extend_forecast
PandasObject.make_training_data = make_training_data
PandasObject.fit_skit_classifier = fit_skit_classifier
PandasObject.fit_classifier = fit_classifier
PandasObject.skit_classify = skit_classify
PandasObject.classify = classify

setattr(pd, 'fetch_yahoo', fetch_yahoo)

__version = 6
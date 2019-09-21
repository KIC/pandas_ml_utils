"""Augment pandas data frames with methods for machine learning"""
__version__ = '0.0.2'

from .pandas_extensions import *
from .classifier import *
from .lazy_dataframe import *
from .fetch_yahoo import *
from .data_objects import *
from .classifier_models import *
from .features_and_Labels import *
from pandas.core.base import PandasObject


# add functions to pandas
PandasObject.hashable = hashable
PandasObject.inner_join = inner_join
PandasObject.drop_re = drop_re
PandasObject.add_apply = add_apply
PandasObject.shift_inplace = shift_inplace
PandasObject.extend_forecast = extend_forecast
PandasObject.make_training_data = make_training_data
PandasObject.fit_classifier = fit_classifier
PandasObject.classify = classify
PandasObject.backtest = backtest

setattr(pd, 'fetch_yahoo', fetch_yahoo)


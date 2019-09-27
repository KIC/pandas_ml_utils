"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.0.5'

from .pandas_utils_extension import *
from .wrappers.lazy_dataframe import *
from .datafetching.fetch_yahoo import *
from .model.models import *
from .model.features_and_Labels import *
from .classification.summary import *
from .classification.classifier import *
from .regression.regressor import *
from .train_test_data import *
from pandas.core.base import PandasObject


# add functions to pandas
# general utility functions
PandasObject.inner_join = inner_join
PandasObject.drop_re = drop_re
PandasObject.add_apply = add_apply
PandasObject.shift_inplace = shift_inplace
PandasObject.extend_forecast = extend_forecast
PandasObject.make_training_data = make_training_data

# classification functions
PandasObject.fit_classifier = fit_classifier
PandasObject.classify = classify
PandasObject.backtest_classifier = backtest_classifier

# regression functions
PandasObject.fit_regressor = fit_regressor
PandasObject.backtest_regressor = backtest_regressor
PandasObject.regress = regress

# data fetcher
setattr(pd, 'fetch_yahoo', fetch_yahoo)


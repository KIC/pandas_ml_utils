"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.0.11'

from .pandas_utils_extension import *
from .analysis.correlation_analysis import *
from .wrappers.lazy_dataframe import *
from .datafetching.fetch_yahoo import *
from .model.models import *
from .model.features_and_Labels import *
from .model.selection import *
from .classification.summary import *
from .classification.classifier import *
from .reinforcement.agent import *
from .regression.regressor import *
from .train_test_data import *
from pandas.core.base import PandasObject


# add functions to pandas
# general utility functions
PandasObject.inner_join = inner_join
PandasObject.drop_re = drop_re
PandasObject.drop_zero_or_nan = drop_zero_or_nan
PandasObject.add_apply = add_apply
PandasObject.shift_inplace = shift_inplace
PandasObject.extend_forecast = extend_forecast
PandasObject.make_training_data = make_training_data

# feature selection
PandasObject.plot_correlation_matrix = plot_correlation_matrix
PandasObject.filtration = filtration

# classification functions
PandasObject.fit_classifier = fit_classifier
PandasObject.classify = classify
PandasObject.backtest_classifier = backtest_classifier

# regression functions
PandasObject.fit_regressor = fit_regressor
PandasObject.backtest_regressor = backtest_regressor
PandasObject.regress = regress

# reinforcement learning
PandasObject.fit_agent = fit_agent
PandasObject.backtest_agent = backtest_agent
PandasObject.agent_take_action = agent_take_action

# data fetcher
setattr(pd, 'fetch_yahoo', fetch_yahoo)
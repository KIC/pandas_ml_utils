"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.0.16'

# imports to provide functionality via root import like import pandas_ml_utils as pmu; pmu.XY
from .pandas_utils_extension import *
from .model.models import Model, SkitModel, KerasModel, MultiModel
from .train_test_data import make_training_data, reshape_rnn_as_ar
from .wrappers.lazy_dataframe import LazyDataFrame

# imports only used to augment pandas classes
from .analysis.correlation_analysis import plot_correlation_matrix
from .datafetching.fetch_yahoo import fetch_yahoo
from .model.features_and_Labels import FeaturesAndLabels
from .analysis.selection import feature_selection
from .classification.classifier import fit_classifier, backtest_classifier, classify
from .reinforcement.agent import fit_agent, backtest_agent, agent_take_action
from .regression.regressor import fit_regressor, backtest_regressor, regress
from pandas.core.base import PandasObject
import pandas as pd


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
PandasObject.feature_selection = feature_selection

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

__doc__ = """
The main concept is to extend pandas DataFrame objects such that you can apply any statistical or machine learning
model directly to the DataFrame.

* feature selection
   - :code:`df.plot_correlation_matrix()`
   - :code:`df.feature_selection()`

* classification
   - :code:`df.fit_classifier(model)`
   - :code:`df.classify(model)`
   - :code:`df.backtest_classifier(model)`

* regression functions
   - :code:`df.fit_regressor(model)`
   - :code:`df.backtest_regressor(model)`
   - :code:`df.regress(model)`

* reinforcement learning
   - :code:`df.fit_agent(model)`
   - :code:`df.backtest_agent(model)`
   - :code:`df.agent_take_action(model)`
  
Where a model is composed of a ML :class:`.Model` and a :class:`.FeaturesAndLabels` object. Every `fit_` returns a 
:class:`.Fit` which provides a :class:`.Summary` and a :code:`.save_model('./models/super.model')` method. Models can
be loaded back via :code:`Model.load('./models/super.model')`.
"""
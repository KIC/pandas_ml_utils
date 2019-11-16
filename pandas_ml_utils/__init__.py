"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.0.15'

from .pandas_utils_extension import *
from .analysis.correlation_analysis import *
from .wrappers.lazy_dataframe import *
from .datafetching.fetch_yahoo import *
from .model.models import *
from .model.features_and_Labels import *
from pandas_ml_utils.analysis.selection import *
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
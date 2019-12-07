"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.0.18'

# imports to provide functionality via root import like import pandas_ml_utils as pmu; pmu.XY
from pandas_ml_utils.pandas_utils_extension import *
from pandas_ml_utils.model.models import Model, SkitModel, KerasModel, MultiModel
from pandas_ml_utils.wrappers.lazy_dataframe import LazyDataFrame
from pandas_ml_utils.model.features_and_labels.features_and_labels import FeaturesAndLabels

# imports only used to augment pandas classes
from pandas_ml_utils.analysis.correlation_analysis import plot_correlation_matrix
from pandas_ml_utils.datafetching.fetch_yahoo import fetch_yahoo
from pandas_ml_utils.model.fitting.fitter import fit, predict, backtest
from pandas_ml_utils.analysis.selection import feature_selection
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

# feature selection
PandasObject.plot_correlation_matrix = plot_correlation_matrix
PandasObject.feature_selection = feature_selection

# new we just have one fit method
PandasObject.fit = fit
PandasObject.predict = predict
PandasObject.backtest = backtest

# data fetcher
setattr(pd, 'fetch_yahoo', fetch_yahoo)

__doc__ = """
The main concept is to extend pandas DataFrame objects such that you can apply any statistical or machine learning
model directly to the DataFrame.

* feature selection
   - :code:`df.plot_correlation_matrix()`
   - :code:`df.feature_selection()`

* fitting, testing and using models
   - :code:`df.fit(model)`
   - :code:`df.backtest(model)`
   - :code:`df.predict(model)`

  
Where a model is composed of a ML :class:`.Model` and a :class:`.FeaturesAndLabels` object. The `fit` method returns a 
:class:`pandas_ml_utils.model.fitting.fit.Fit` which provides a :class:`.Summary` and a :code:`.save_model('./models/super.model')` method. Models can
be loaded back via :code:`Model.load('./models/super.model')`.
"""
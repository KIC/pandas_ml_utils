"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.0.27'

import logging

from pandas.core.base import PandasObject as _PandasObject

import numpy as np
import pandas as pd
import pandas_ml_utils.pandas_utils_extension as _df_ext
from pandas_ml_utils.analysis.correlation_analysis import plot_correlation_matrix as _plot_correlation_matrix
from pandas_ml_utils.analysis.selection import feature_selection as _feature_selection
from pandas_ml_utils.datafetching import fetch_cryptocompare_hourly as _fetch_cryptocompare_hourly, \
    fetch_cryptocompare_daily as _fetch_cryptocompare_daily, fetch_yahoo as _fetch_yahoo
from pandas_ml_utils.model.features_and_labels.features_and_labels import FeaturesAndLabels
from pandas_ml_utils.model.fitting.fitter import fit as _fit, predict as _predict, backtest as _backtest, \
    features_and_label_extractor as _features_and_label_extractor
from pandas_ml_utils.model.models import Model, SkModel, KerasModel, MultiModel
from pandas_ml_utils.wrappers.lazy_dataframe import LazyDataFrame

# log provided classes
_log = logging.getLogger(__name__)
_log.debug(f"available {Model} classes {[SkModel, KerasModel, MultiModel]}")
_log.debug(f"available other classes {[LazyDataFrame, FeaturesAndLabels]}")
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")

# add functions to pandas
# general utility functions
_PandasObject.cloc2 = _df_ext.cloc2
_PandasObject.inner_join = _df_ext.inner_join
_PandasObject.drop_re = _df_ext.drop_re
_PandasObject.drop_zero_or_nan = _df_ext.drop_zero_or_nan
_PandasObject.add_apply = _df_ext.add_apply
_PandasObject.shift_inplace = _df_ext.shift_inplace
_PandasObject.extend_forecast = _df_ext.extend_forecast

# feature selection
_PandasObject.plot_correlation_matrix = _plot_correlation_matrix
_PandasObject.feature_selection = _feature_selection

# provide fit, predict and backtest method
_PandasObject.fit = _fit
_PandasObject.predict = _predict
_PandasObject.backtest = _backtest

# also provide the plan features and labels extractor
_PandasObject.features_and_label_extractor = _features_and_label_extractor

# data fetcher
setattr(pd, 'fetch_yahoo', _fetch_yahoo)
setattr(pd, 'fetch_cryptocompare_daily', _fetch_cryptocompare_daily)
setattr(pd, 'fetch_cryptocompare_hourly', _fetch_cryptocompare_hourly)

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

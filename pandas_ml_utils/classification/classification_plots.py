import inspect
import logging
import sys
from typing import List, Tuple, Callable, Iterable, Dict, Union
from matplotlib import pyplot as plt

import seaborn as sns
import numpy as np
import pandas as pd


log = logging.getLogger(__name__)


def plot_forecast_heatmap(df: pd.DataFrame,
                          multi_model: object,
                          parameter_as_column: str):

    # we need a data frame with the target values as row index and the forecast periods as columns
    # if we have one data frame (which is effectively one row) per "target" we just need to concatenete them
    # and join the same frames as new column for each forecast period
    pass
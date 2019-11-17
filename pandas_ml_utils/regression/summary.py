import logging
import sys
from typing import Tuple, Union, Dict

import numpy as np
import pandas as pd

from ..model.summary import Summary

_log = logging.getLogger(__name__)


class RegressionSummary(Summary):

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # TODO add some statistics and fix loss is always None at the moment
    #  mse, r2, ... pvalue?
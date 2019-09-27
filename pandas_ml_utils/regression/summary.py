import logging
import sys
from typing import Tuple, Union

import numpy as np
import pandas as pd

from ..model.summary import Summary

log = logging.getLogger(__name__)


class RegressionSummary(Summary):

    def __init__(self,
                 y_true: np.ndarray,
                 y_prediction: np.ndarray,
                 index: np.ndarray,
                 loss: pd.Series = None):
        self.y_true = y_true
        self.y_prediction = y_prediction.ravel() if len(y_prediction.shape) > 1 else y_prediction
        self.index = index
        self.loss = loss

    # TODO add some statistics
import logging
import sys
from typing import Tuple, Union, Dict

import numpy as np
import pandas as pd

from ..model.summary import Summary

log = logging.getLogger(__name__)


class RegressionSummary(Summary):

    def __init__(self,
                 y_true: np.ndarray,
                 y_predictions: Dict[str, np.ndarray],
                 index: np.ndarray,
                 loss: pd.Series = None):
        self.y_true = y_true
        self.y_prediction = {target: (y.ravel() if len(y.shape) > 1 else y) for target, y in y_predictions.items()}
        self.index = index
        self.loss = loss

    # TODO add some statistics and fix loss is always None at the moment
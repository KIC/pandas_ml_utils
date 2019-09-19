import inspect
import logging
import sys
from typing import List, Tuple, Callable, Iterable, Dict, Union
from matplotlib import pyplot as plt

import seaborn as sns
import numpy as np
import pandas as pd

from .classifier_models import Model

log = logging.getLogger(__name__)


def plot_forecast_heatmap(df: pd.DataFrame,
                          model: Model):

    pass
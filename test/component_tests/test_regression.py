import logging
import os
import unittest

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

import pandas_ml_utils as pdu

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test.csv")


class RegressionTest(unittest.TestCase):
    pass
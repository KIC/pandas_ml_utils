import logging
import unittest

import pandas as pd
from test.config import TEST_FILE

from pandas_ml_utils.analysis.correlation_analysis import _sort_correlation

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class ComponentTest(unittest.TestCase):

    def test_correlation_matrix(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')[["spy_Open", "spy_High", "spy_Low", "spy_Close"]]

        """when"""
        corr = _sort_correlation(df.corr(), recursive=True)

        """then"""
        self.assertListEqual(corr.columns.tolist(), ["spy_Open", "spy_High", "spy_Close", "spy_Low"])


import logging
import unittest

import pandas as pd

from pandas_ml_utils.summary.summary import Summary

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class SummaryTest(unittest.TestCase):

    def test_summary(self):
        """given"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})

        """when"""
        s = Summary(df)

        """then"""
        self.assertEqual(len(s._repr_html_()), 635)
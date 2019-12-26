import logging
from unittest import TestCase

import pandas as pd
import pandas_ml_utils as pmu

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.info(f"{pmu.__version__}")


class TestFetchYahoo(TestCase):

    def test__with_prefix(self):
        """when"""
        df = pd.fetch_yahoo(spy="spy", vix="^vix", multi_index=False)

        """then"""
        self.assertListEqual(df.columns.tolist(), ['spy_Open', 'spy_High', 'spy_Low', 'spy_Close', 'spy_Volume', 'spy_Dividends', 'spy_Stock Splits', 'vix_Open', 'vix_High', 'vix_Low', 'vix_Close', 'vix_Volume', 'vix_Dividends', 'vix_Stock Splits'])

    def test__with_multiindex(self):
        """when"""
        df = pd.fetch_yahoo(spy="spy", vix="^vix", multi_index=True)

        """then"""
        self.assertListEqual(df.columns.tolist(), [('spy', 'Open'), ('spy', 'High'), ('spy', 'Low'), ('spy', 'Close'), ('spy', 'Volume'), ('spy', 'Dividends'), ('spy', 'Stock Splits'), ('vix', 'Open'), ('vix', 'High'), ('vix', 'Low'), ('vix', 'Close'), ('vix', 'Volume'), ('vix', 'Dividends'), ('vix', 'Stock Splits')])

    def test__with_multiindex_args(self):
        """when"""
        df = pd.fetch_yahoo("spy", "^vix", multi_index=True)

        """then"""
        self.assertListEqual(df.columns.tolist(), [('spy', 'Open'), ('spy', 'High'), ('spy', 'Low'), ('spy', 'Close'), ('spy', 'Volume'), ('spy', 'Dividends'), ('spy', 'Stock Splits'), ('^vix', 'Open'), ('^vix', 'High'), ('^vix', 'Low'), ('^vix', 'Close'), ('^vix', 'Volume'), ('^vix', 'Dividends'), ('^vix', 'Stock Splits')])

    def test__single(self):
        """when"""
        df = pd.fetch_yahoo("spy")

        """then"""
        self.assertListEqual(df.columns.tolist(), ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'])

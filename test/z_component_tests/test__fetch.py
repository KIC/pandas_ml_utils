import logging
import unittest

import pandas as pd

import pandas_ml_utils as pdu

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
print(pdu.__version__)


class ComponentTest(unittest.TestCase):

    def test_fetch_yahoo(self):
        """when"""
        df = pd.fetch_yahoo(spy="SPY").tail()
        print(df.columns)

        """then"""
        self.assertTrue(df["spy_Close"].sum() > 0)


    def test_crypto_compare(self):
        """when"""
        df = pd.fetch_cryptocompare_daily(coin="BTC", limit=None).tail()
        print(df.columns)

        """then"""
        self.assertTrue(df["close"].sum() > 0)


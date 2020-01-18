import unittest
import logging
import pandas as pd
import pandas_ml_utils.extern.cryptocompare as cc

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestCryptoCompare(unittest.TestCase):

    @unittest.skip("only test if error, takes very long")
    def test_download_day(self):
        """when"""
        data = cc.get_historical_price_day("BTC", limit=None)["Data"]

        """then"""
        print(f"\n{len(data)}")
        self.assertTrue(len(data) > 2000)
        self.assertEqual(pd.DataFrame(data).duplicated("time").astype(int).sum(), 0)

    @unittest.skip("only test if error, takes very long")
    def test_bach_load_hour(self):
        data = cc.get_historical_price_hour("BTC", limit=None)["Data"]

        """then"""
        print(f"\n{len(data)}")
        self.assertTrue(len(data) > 300100)
        self.assertEqual(pd.DataFrame(data).duplicated("time").astype(int).sum(), 0)

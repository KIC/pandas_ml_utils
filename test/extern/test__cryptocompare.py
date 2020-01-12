import unittest
import pandas_ml_utils.extern.cryptocompare as cc


class TestCryptoCompare(unittest.TestCase):

    def test_download(self):
        """when"""
        data = cc.get_historical_price_day("BTC", limit=None)["Data"]

        """then"""
        print(f"\n{len(data)}")
        self.assertTrue(len(data) > 2000)
import unittest
import logging
import pandas as pd
import pytest

import pandas_ml_utils.extern.cryptocompare as cc
from pandas_ml_utils.extern.cryptocompare import TIME
from pandas_ml_utils.datafetching.fetch_cryptocompare import _data_to_frame

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
DEBUG = False


class TestCryptoCompare(unittest.TestCase):

    @pytest.mark.skipif(DEBUG is False, reason="test only if needed due to rate limit")
    def test_download_day(self):
        """when"""
        data = cc.get_historical_price_day("BTC", limit=None)["Data"]
        df = _data_to_frame(data)

        """then"""
        print(f"\n{len(data)}")
        self.assertTrue(len(data) > 2000)
        self.assertEqual(len(data), len(df))
        self.assertEqual(pd.DataFrame(data).duplicated("time").astype(int).sum(), 0)

    @pytest.mark.skipif(DEBUG is False, reason="test only if needed due to rate limit")
    def test_bach_load_hour(self):
        data = cc.get_historical_price_hour("BTC", limit=None)["Data"]
        youngest_ts = min([r[TIME] for r in data])

        """then"""
        print(f"\n{len(data)}, {youngest_ts}")
        self.assertTrue(youngest_ts >= 1270810800 - 60 * 60 * 24)
        self.assertTrue(len(data) > 86000)
        self.assertEqual(pd.DataFrame(data).duplicated("time").astype(int).sum(), 0)

import logging
import os
import unittest

import pandas as pd
from sklearn.neural_network import MLPClassifier

import pandas_ml_utils as pdu

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test.csv")


class LazyDataFrameTest(unittest.TestCase):

    def test_fit_and_co(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date').tail(100)
        ldf = pdu.LazyDataFrame(
            df,
            sma=lambda f: f["vix_Close"].rolling(2).mean(),
            label=lambda f: f["spy_Close"] > f["spy_Open"]
        )
        model = pdu.SkitModel(
            MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
            pdu.FeaturesAndLabels(["sma"], ["label"])
        )

        """when"""
        fit = ldf.fit(model)
        bt = ldf.backtest(fit.model)
        p = ldf.predict(fit.model)

        """then"""
        self.assertEqual(len(fit.test_summary.df), 40)
        self.assertEqual(len(bt.df), 100 - 1)
        self.assertEqual(len(p), 100 - 1)

    def test_pre_process_and_fit_and_co(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date').tail(100)
        model = pdu.SkitModel(
            MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
            pdu.FeaturesAndLabels(
                ["sma"],
                ["label"],
                pre_processor=lambda _df, _: pdu.LazyDataFrame(
                    _df,
                    sma=lambda f: f["vix_Close"].rolling(2).mean(),
                    label=lambda f: f["spy_Close"] > f["spy_Open"]
                )
            )
        )

        """when"""
        fit = df.fit(model)
        bt = df.backtest(fit.model)
        p = df.predict(fit.model)

        """then"""
        self.assertEqual(len(fit.test_summary.df), 40)
        self.assertEqual(len(bt.df), 100 - 1)
        self.assertEqual(len(p), 100 - 1)


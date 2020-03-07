import logging
import unittest

from sklearn.neural_network import MLPClassifier
from test.config import TEST_FILE

from pandas_ml_utils import pd, LazyDataFrame, SkModel, FeaturesAndLabels

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class LazyDataFrameTest(unittest.TestCase):

    def test_fit_and_co(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date').tail(100)
        ldf = LazyDataFrame(
            df,
            sma=lambda f: f["vix_Close"].rolling(2).mean(),
            label=lambda f: f["spy_Close"] > f["spy_Open"]
        )
        model = SkModel(
            MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
            FeaturesAndLabels(["sma"], ["label"])
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
        model = SkModel(
            MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
            FeaturesAndLabels(
                ["sma"],
                ["label"],
                pre_processor=lambda _df: LazyDataFrame(
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


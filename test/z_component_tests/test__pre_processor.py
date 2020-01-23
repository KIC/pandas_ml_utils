import logging
import unittest

import pandas as pd
from sklearn.neural_network import MLPClassifier

import pandas_ml_utils as pdu
from pandas_ml_utils.constants import *
from test.config import TEST_FILE

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class PreprocessorTest(unittest.TestCase):

    def test_pre_processor(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')

        """when"""
        fit = df.fit(
            pdu.SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), alpha=0.001, random_state=42),
                pdu.FeaturesAndLabels(features=['feature'], labels=['label'],
                                      gross_loss=lambda df: df["spy_Close"] - df["spy_Open"],
                                      pre_processor=lambda _df: pdu.LazyDataFrame(
                                          _df,
                                          feature=lambda f: f["vix_Close"].rolling(2).mean(),
                                          label=lambda f: (f["spy_Close"].shift(1) > f["spy_Open"]).shift(-1)).to_dataframe())),
            test_size=0.4,
            test_validate_split_seed=42)

        bt = df.backtest(fit.model)
        p = df.predict(fit.model, 2)

        """then fit"""
        self.assertListEqual(fit.test_summary.df.columns.tolist(), [(PREDICTION_COLUMN_NAME, 'label'), (LABEL_COLUMN_NAME, 'label'), (GROSS_LOSS_COLUMN_NAME, GROSS_LOSS_COLUMN_NAME)])

        """ and backtest"""
        self.assertListEqual(bt.df.columns.tolist(), [(PREDICTION_COLUMN_NAME, 'label'), (LABEL_COLUMN_NAME, 'label'), (GROSS_LOSS_COLUMN_NAME, GROSS_LOSS_COLUMN_NAME), *[(SOURCE_COLUMN_NAME, c) for c in df.columns], (SOURCE_COLUMN_NAME, "feature"), (SOURCE_COLUMN_NAME, "label")])
        self.assertEqual(bt.df.index[-1], "2019-09-13")

        """ and prediction"""
        self.assertListEqual(p.columns.tolist(), [(PREDICTION_COLUMN_NAME, 'label')])
        self.assertEqual(p.index[-1], "2019-09-16")
        self.assertEqual(len(p), 2)

import logging
import unittest

import pandas as pd

import pandas_ml_utils as pdu
from pandas_ml_utils.analysis.correlation_analysis import _sort_correlation
from pandas_ml_utils.model.features_and_labels.features_and_labels_extractor import FeatureTargetLabelExtractor
from pandas_ml_utils.model.fitting.train_test_data import make_training_data
from test.config import TEST_FILE

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class ComponentTest(unittest.TestCase):

    def test_fetch_yahoo(self):
        """when"""
        df = pd.fetch_yahoo(spy="SPY").tail()

        """then"""
        self.assertTrue(df["spy_Close"].sum() > 0)

    def test_correlation_matrix(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')[["spy_Open", "spy_High", "spy_Low", "spy_Close"]]

        """when"""
        corr = _sort_correlation(df.corr(), recursive=True)

        """then"""
        self.assertListEqual(corr.columns.tolist(), ["spy_Open", "spy_High", "spy_Close", "spy_Low"])

    def test_make_train_data(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df['label'] = df["spy_Close"] > df["spy_Open"]

        """when"""
        x_train, x_test, y_train, y_test, index_train, index_test = \
            make_training_data(
                FeatureTargetLabelExtractor(
                    df,
                    pdu.FeaturesAndLabels(features=['vix_Close'], labels=['label'], feature_lags=[0, 1, 2])))

        """then"""
        self.assertEqual(x_train.shape, (4022, 3, 1))
        self.assertEqual(y_train.shape, (4022, 1))

        self.assertEqual(x_test.shape, (2682, 3, 1))
        self.assertEqual(y_test.shape, (2682, 1))

        self.assertEqual(len(x_train), len(index_train))
        self.assertEqual(len(x_test), len(index_test))


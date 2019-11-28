import logging
import os
import unittest

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

import pandas_ml_utils as pdu

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test.csv")


class MultiModelTest(unittest.TestCase):

    def test_invalid_multi_model(self):
        """expect"""
        self.assertRaises(ValueError,
                          lambda: pdu.MultiModel(pdu.MultiModel(pdu.SkitModel(MLPClassifier(), pdu.FeaturesAndLabels([],{})))))

    def test_multi_model_binary_classifications(self):
        import talib

        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["sma"] = talib.SMA(df["spy_Close"])
        df["is_above_1.0"] = (df["spy_Close"] / df["sma"]) > 1
        df["is_above_1.2"] = (df["spy_Close"] / df["sma"]) > 1.2

        model = pdu.MultiModel(
            pdu.SkitModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
                pdu.FeaturesAndLabels(features=['vix_Close'],
                                      labels={"a": ["is_above_1.0"], "b": ["is_above_1.2"]},
                                      targets=lambda t, frame: frame["sma"].rename(f"sma {t}"),
                                      loss=lambda _, frame: frame["spy_Close"] - frame["sma"])))

        """when"""
        fit = df.fit(model, test_size=0.4, test_validate_split_seed=42,)
        result = fit.training_summary.df

        """then"""
        print(result.columns.tolist())
        self.assertListEqual(result.columns.tolist(),
                             [('a', 'prediction', 'is_above_1.0'), ('b', 'prediction', 'is_above_1.2'),
                              ('a', 'label', 'is_above_1.0'), ('b', 'label', 'is_above_1.2'),
                              ('a', 'loss', 'a'), ('b', 'loss', 'b'),
                              ('a', 'target', 'sma a'), ('b', 'target', 'sma b')])

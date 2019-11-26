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


class ClassificationTest(unittest.TestCase):

    def test_binary_classification(self):
        import talib

        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["sma"] = talib.SMA(df["spy_Close"])
        df["is_above"] = (df["spy_Close"] / df["sma"]) > 1

        model = pdu.SkitModel(
            MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
            pdu.FeaturesAndLabels(features=['vix_Close'],
                                  labels=("sma", ["is_above"]),
                                  loss=lambda target, frame: frame["spy_Close"] - frame["sma"]))

        """when"""
        fit = df.fit(model, test_size=0.4, test_validate_split_seed=42)
        result = fit.training_summary.target_data

        """then"""
        self.assertListEqual(list(result.keys()), [("sma", )])
        self.assertListEqual(result[("sma", )].columns.tolist(), [('prediction', 'is_above'), ('label', 'is_above'), ('loss', 'loss')])
        self.assertEqual(len(result[("sma", )]), 4023)

    def test_multiple_classifications(self):
        # FIXME implement this
        from pandas_ml_utils.model.features_and_labels_utils.target_encoder import OneHotEncodedTargets
        import talib

        """given"""
        df = pd.read_csv(f'{__name__}.csv', index_col='Date')
        df["sma"] = talib.SMA(df["spy_Close"])
        df["label"] = df["spy_Close"] / df["sma"] -1

        model = pdu.SkitModel(
            MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
            pdu.FeaturesAndLabels(features=['vix_Close'],
                                  labels=OneHotEncodedTargets("label", np.linspace(-0.1, 0.1, 3, endpoint=True))))

        """when"""
        fit = df.fit_classifier(model, test_size=0.4, test_validate_split_seed=42,)
        result = fit.training_summary.target_data

        """then"""
        self.assertListEqual(list(result.keys()), [("sma",)])
        self.assertListEqual(result[("sma",)].columns.tolist(), [('prediction', 'is_above'), ('label', 'is_above'), ('loss', 'loss')])
        self.assertEqual(len(result[("sma",)]), 4023)

    def test_multi_model_multiple_classifications(self):
        # FIXME implement this
        from pandas_ml_utils.model.features_and_labels_utils.target_encoder import OneHotEncodedTargets
        import talib

        df = pd.read_csv(f'{__name__}.csv', index_col='Date')
        df["sma"] = talib.SMA(df["spy_Close"])
        df["label_1"] = df["spy_Close"] / df["sma"] - 1
        df["label_2"] = df["spy_Close"] / df["sma"] - 2

        model = pdu.MultiModel(
            pdu.SkitModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
                          pdu.FeaturesAndLabels(features=['vix_Close'],
                                                labels={f"-{t}": OneHotEncodedTargets("label", np.linspace(-0.1, 0.1, 3, endpoint=True)) for t in range(1, 3)})))

        fit = df.fit_classifier(model, test_size=0.4, test_validate_split_seed=42,)

        self.assertTrue(False)



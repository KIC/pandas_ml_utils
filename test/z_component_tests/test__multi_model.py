import logging
import os
import unittest

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

import pandas_ml_utils as pdu
from pandas_ml_utils.model.features_and_labels.target_encoder import OneHotEncodedTargets
from test.utils import SMA

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test.csv")


class MultiModelTest(unittest.TestCase):

    def test_invalid_multi_model(self):
        """expect"""
        self.assertRaises(ValueError,
                          lambda: pdu.MultiModel(pdu.MultiModel(pdu.SkitModel(MLPClassifier(), pdu.FeaturesAndLabels([],{})))))

    def test_multi_model_binary_classifications(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["sma"] = SMA(df["spy_Close"])
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
        fit_summary_df = fit.training_summary.df
        bt_summary_df = df.backtest(fit.model).df
        predict_df = df.predict(fit.model, tail=1)

        """then"""
        self.assertListEqual(fit_summary_df.columns.tolist(),
                             [('a', 'prediction', 'is_above_1.0'), ('b', 'prediction', 'is_above_1.2'),
                              ('a', 'label', 'is_above_1.0'), ('b', 'label', 'is_above_1.2'),
                              ('a', 'loss', 'a'), ('b', 'loss', 'b'),
                              ('a', 'target', 'sma a'), ('b', 'target', 'sma b')])

        self.assertListEqual(bt_summary_df.columns.tolist(), fit_summary_df.columns.tolist())

        self.assertListEqual(predict_df.columns.tolist(),
                             [('a', 'prediction', 'is_above_1.0'), ('b', 'prediction', 'is_above_1.2'),
                              ('a', 'target', 'sma a'), ('b', 'target', 'sma b')])

    def test_multi_model_multi_class_classifications(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["sma"] = SMA(df["spy_Close"])
        df["is_above_1.0"] = (df["spy_Close"] / df["sma"]) + 1
        df["is_above_1.2"] = (df["spy_Close"] / df["sma"]) + 2

        model = pdu.MultiModel(
            pdu.SkitModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
                pdu.FeaturesAndLabels(features=['vix_Close'],
                                      labels={"1": OneHotEncodedTargets("is_above_1.0", np.linspace(-0.1, 0.1, 5, endpoint=True) + 1),
                                              "2": OneHotEncodedTargets("is_above_1.2", np.linspace(-0.1, 0.1, 5, endpoint=True) + 2)},
                                      targets=lambda t, frame: (frame["sma"] + int(t)).rename(f"sma {t}"),
                                      loss=lambda _, frame: frame["spy_Close"] - frame["sma"])))

        """when"""
        fit = df.fit(model, test_size=0.4, test_validate_split_seed=42,)
        fit_summary_df = fit.training_summary.df
        bt_summary_df = df.backtest(fit.model).df
        predict_df = df.predict(fit.model, tail=1)

        """then"""
        self.assertListEqual(fit_summary_df.columns.tolist(),
                             [('1', 'prediction', 'is_above_1.0 #0'), ('1', 'prediction', 'is_above_1.0 #1'), ('1', 'prediction', 'is_above_1.0 #2'), ('1', 'prediction', 'is_above_1.0 #3'),
                              ('2', 'prediction', 'is_above_1.2 #0'), ('2', 'prediction', 'is_above_1.2 #1'), ('2', 'prediction', 'is_above_1.2 #2'), ('2', 'prediction', 'is_above_1.2 #3'),
                              ('1', 'label', 'is_above_1.0 #0'), ('1', 'label', 'is_above_1.0 #1'), ('1', 'label', 'is_above_1.0 #2'), ('1', 'label', 'is_above_1.0 #3'),
                              ('2', 'label', 'is_above_1.2 #0'), ('2', 'label', 'is_above_1.2 #1'), ('2', 'label', 'is_above_1.2 #2'), ('2', 'label', 'is_above_1.2 #3'),
                              ('1', 'loss', '1'), ('2', 'loss', '2'),
                              ('1', 'target', 'sma 1'), ('2', 'target', 'sma 2')])

        self.assertListEqual(bt_summary_df.columns.tolist(), fit_summary_df.columns.tolist())

        self.assertListEqual(predict_df.columns.tolist(),
                             [('1', 'prediction', 'is_above_1.0 #0'), ('1', 'prediction', 'is_above_1.0 #1'), ('1', 'prediction', 'is_above_1.0 #2'), ('1', 'prediction', 'is_above_1.0 #3'),
                              ('2', 'prediction', 'is_above_1.2 #0'), ('2', 'prediction', 'is_above_1.2 #1'), ('2', 'prediction', 'is_above_1.2 #2'), ('2', 'prediction', 'is_above_1.2 #3'),
                              ('1', 'target', 'sma 1'), ('2', 'target', 'sma 2')])


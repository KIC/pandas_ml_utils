import logging
import unittest

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

import pandas_ml_utils as pdu
from pandas_ml_utils.constants import *
from pandas_ml_utils.model.features_and_labels.target_encoder import OneHotEncodedTargets
from test.config import TEST_FILE
from test.utils import SMA

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class MultiModelTest(unittest.TestCase):
    maxDiff = None

    def test_invalid_multi_model(self):
        """expect"""
        self.assertRaises(ValueError,
                          lambda: pdu.MultiModel(pdu.MultiModel(pdu.SkModel(MLPClassifier(), pdu.FeaturesAndLabels([], {})))))

    def test_multi_model_binary_classifications(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["sma"] = SMA(df["spy_Close"])
        df["is_above_1.0"] = (df["spy_Close"] / df["sma"]) > 1
        df["is_above_1.2"] = (df["spy_Close"] / df["sma"]) > 1.2

        model = pdu.MultiModel(
            pdu.SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
                pdu.FeaturesAndLabels(features=['vix_Close'],
                                      labels={"a": ["is_above_1.0"], "b": ["is_above_1.2"]},
                                      targets=lambda frame, t: frame["sma"].rename(f"sma {t}"),
                                      gross_loss=lambda frame: frame["spy_Close"] - frame["sma"])))

        """when"""
        fit = df.fit(model, test_size=0.4, test_validate_split_seed=42,)
        fit_summary_df = fit.training_summary.df
        bt_summary_df = df.backtest(fit.model).df
        predict_df = df.predict(fit.model, tail=1)

        """then"""
        self.assertListEqual(fit_summary_df.columns.tolist(),
                             [('a', PREDICTION_COLUMN_NAME, 'is_above_1.0'), ('b', PREDICTION_COLUMN_NAME, 'is_above_1.2'),
                              ('a', LABEL_COLUMN_NAME, 'is_above_1.0'), ('b', LABEL_COLUMN_NAME, 'is_above_1.2'),
                              ('a', GROSS_LOSS_COLUMN_NAME, 'a'), ('b', GROSS_LOSS_COLUMN_NAME, 'b'),
                              ('a', TARGET_COLUMN_NAME, 'sma a'), ('b', TARGET_COLUMN_NAME, 'sma b')])

        self.assertListEqual(predict_df.columns.tolist(),
                             [('a', PREDICTION_COLUMN_NAME, 'is_above_1.0'), ('b', PREDICTION_COLUMN_NAME, 'is_above_1.2'),
                              ('a', TARGET_COLUMN_NAME, 'sma a'), ('b', TARGET_COLUMN_NAME, 'sma b')])

        self.assertEqual(bt_summary_df.shape, (6706, 20))

    def test_multi_model_multi_class_classifications(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["sma"] = SMA(df["spy_Close"])
        df["is_above_1.0"] = (df["spy_Close"] / df["sma"]) + 1
        df["is_above_1.2"] = (df["spy_Close"] / df["sma"]) + 2

        model = pdu.MultiModel(
            pdu.SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
                pdu.FeaturesAndLabels(features=['vix_Close'],
                                      labels={"1": OneHotEncodedTargets("is_above_1.0", np.linspace(-0.1, 0.1, 5, endpoint=True) + 1),
                                              "2": OneHotEncodedTargets("is_above_1.2", np.linspace(-0.1, 0.1, 5, endpoint=True) + 2)},
                                      targets=lambda frame, t: (frame["sma"] + int(t)).rename(f"sma {t}"),
                                      gross_loss=lambda frame: frame["spy_Close"] - frame["sma"])))

        """when"""
        fit = df.fit(model, test_size=0.4, test_validate_split_seed=42,)
        fit_summary_df = fit.training_summary.df
        bt_summary_df = df.backtest(fit.model).df
        predict_df = df.predict(fit.model, tail=1)

        """then"""
        print(fit_summary_df.columns.tolist())
        self.assertListEqual(fit_summary_df.columns.tolist(),
                             [('1', 'prediction', '(-inf, 0.95]'), ('1', 'prediction', '(0.95, 1.0]'), ('1', 'prediction', '(1.0, 1.05]'), ('1', 'prediction', '(1.05, inf]'),
                              ('2', 'prediction', '(-inf, 1.95]'), ('2', 'prediction', '(1.95, 2.0]'), ('2', 'prediction', '(2.0, 2.05]'), ('2', 'prediction', '(2.05, inf]'),
                              ('1', 'label', '(-inf, 0.95]'), ('1', 'label', '(0.95, 1.0]'), ('1', 'label', '(1.0, 1.05]'), ('1', 'label', '(1.05, inf]'),
                              ('2', 'label', '(-inf, 1.95]'), ('2', 'label', '(1.95, 2.0]'), ('2', 'label', '(2.0, 2.05]'), ('2', 'label', '(2.05, inf]'),
                              ('1', GROSS_LOSS_COLUMN_NAME, '1'), ('2', GROSS_LOSS_COLUMN_NAME, '2'),
                              ('1', TARGET_COLUMN_NAME, 'sma 1'), ('2', TARGET_COLUMN_NAME, 'sma 2')])

        self.assertListEqual(predict_df.columns.tolist(),
                             [('1', 'prediction', '(-inf, 0.95]'), ('1', 'prediction', '(0.95, 1.0]'), ('1', 'prediction', '(1.0, 1.05]'), ('1', 'prediction', '(1.05, inf]'),
                              ('2', 'prediction', '(-inf, 1.95]'), ('2', 'prediction', '(1.95, 2.0]'), ('2', 'prediction', '(2.0, 2.05]'), ('2', 'prediction', '(2.05, inf]'),
                              ('1', TARGET_COLUMN_NAME, 'sma 1'), ('2', TARGET_COLUMN_NAME, 'sma 2')])

        self.assertEqual(bt_summary_df.shape, (6706, 32))


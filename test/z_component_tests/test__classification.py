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


class ClassificationTest(unittest.TestCase):
    maxDiff = None

    def test_binary_classification(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["sma"] = SMA(df["spy_Close"])
        df["is_above"] = (df["spy_Close"] / df["sma"]) > 1

        model = pdu.SkModel(
            MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
            pdu.FeaturesAndLabels(features=['vix_Close'],
                                  labels=["is_above"],
                                  targets=lambda frame: frame["sma"],
                                  gross_loss=lambda frame: frame["spy_Close"] - frame["sma"]))

        """when"""
        fit = df.fit(model, test_size=0.4, test_validate_split_seed=42)
        fit_summary_df = fit.training_summary.df
        bt_summary_df = df.backtest(fit.model).df
        predict_df = df.predict(fit.model, tail=1)

        """then"""
        self.assertListEqual(fit_summary_df.columns.tolist(), [(PREDICTION_COLUMN_NAME, 'is_above'), (LABEL_COLUMN_NAME, 'is_above'), (GROSS_LOSS_COLUMN_NAME, GROSS_LOSS_COLUMN_NAME), (TARGET_COLUMN_NAME, 'sma')])
        self.assertEqual(len(fit_summary_df), 4023)

        self.assertEqual(bt_summary_df.shape, (6706, 15))

        self.assertListEqual(predict_df.columns.tolist(), [(PREDICTION_COLUMN_NAME, 'is_above'), (TARGET_COLUMN_NAME, 'sma')])

    def test_multi_class_classification(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["sma"] = SMA(df["spy_Close"])
        df["label"] = df["spy_Close"] / df["sma"] -1

        def make_targets(frame):
            space = np.array([-1, -0.05, 0.5, 1])
            res = frame.apply(lambda x: x["sma"] - space, axis=1,
                              result_type='expand')
            res.columns = ["close <0.1", "close <0.05", "close >0", "close >0.05"]
            return res

        model = pdu.SkModel(
            MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
            pdu.FeaturesAndLabels(features=['vix_Close'],
                                  labels=OneHotEncodedTargets("label", np.linspace(-0.1, 0.1, 5, endpoint=True)),
                                  targets=make_targets))

        """when"""
        fit = df.fit(model, test_size=0.4, test_validate_split_seed=42,)
        fit_summary_df = fit.training_summary.df
        bt_summary_df = df.backtest(fit.model).df
        predict_df = df.predict(fit.model, tail=1)

        """then"""
        self.assertEqual(len(fit_summary_df), 4023)
        self.assertListEqual(fit_summary_df.columns.tolist(),
                             [(PREDICTION_COLUMN_NAME, '(-inf, -0.05]'), (PREDICTION_COLUMN_NAME, '(-0.05, 0.0]'), (PREDICTION_COLUMN_NAME,  '(0.0, 0.05000000000000002]'), (PREDICTION_COLUMN_NAME,  '(0.05000000000000002, inf]'),
                              (LABEL_COLUMN_NAME, '(-inf, -0.05]'), (LABEL_COLUMN_NAME, '(-0.05, 0.0]'), (LABEL_COLUMN_NAME,  '(0.0, 0.05000000000000002]'), (LABEL_COLUMN_NAME,  '(0.05000000000000002, inf]'),
                              (TARGET_COLUMN_NAME, 'close <0.1'), (TARGET_COLUMN_NAME, 'close <0.05'), (TARGET_COLUMN_NAME, 'close >0'), (TARGET_COLUMN_NAME, 'close >0.05')])

        self.assertListEqual(predict_df.columns.tolist(),
                             [(PREDICTION_COLUMN_NAME, '(-inf, -0.05]'), (PREDICTION_COLUMN_NAME, '(-0.05, 0.0]'), (PREDICTION_COLUMN_NAME,  '(0.0, 0.05000000000000002]'), (PREDICTION_COLUMN_NAME,  '(0.05000000000000002, inf]'),
                              (TARGET_COLUMN_NAME, 'close <0.1'), (TARGET_COLUMN_NAME, 'close <0.05'), (TARGET_COLUMN_NAME, 'close >0'), (TARGET_COLUMN_NAME, 'close >0.05')])

        self.assertEqual(bt_summary_df.shape, (6706, 23))

    def test_target_classification(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["sma"] = SMA(df["spy_Close"])
        df["is_above_1.0"] = (df["spy_Close"] / df["sma"]) > 1
        df["is_above_1.2"] = (df["spy_Close"] / df["sma"]) > 1.2

        model = pdu.SkModel(
            MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
            pdu.FeaturesAndLabels(features=['vix_Close'],
                                  labels={"a": ["is_above_1.0"], "b": ["is_above_1.2"]}))

        """when"""
        fit = df.fit(model, test_size=0.4, test_validate_split_seed=42)
        fit_summary_df = fit.training_summary.df
        bt_summary_df = df.backtest(fit.model).df
        predict_df = df.predict(fit.model, tail=1)

        """then"""
        self.assertListEqual(fit_summary_df.columns.tolist(),
                             [('a', PREDICTION_COLUMN_NAME, 'is_above_1.0'), ('b', PREDICTION_COLUMN_NAME, 'is_above_1.2'),
                              ('a', LABEL_COLUMN_NAME, 'is_above_1.0'), ('b', LABEL_COLUMN_NAME, 'is_above_1.2')])

        self.assertListEqual(predict_df.columns.tolist(),
                             [('a', PREDICTION_COLUMN_NAME, 'is_above_1.0'), ('b', PREDICTION_COLUMN_NAME, 'is_above_1.2')])

        self.assertEqual(bt_summary_df.shape, (6706, 16))

    def test_lagged_classification(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["sma"] = SMA(df["spy_Close"])
        df["is_above"] = (df["spy_Close"] / df["sma"]) > 1

        model = pdu.SkModel(
            MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
            pdu.FeaturesAndLabels(features=['vix_Close'],
                                  feature_lags=[0, 1, 2],
                                  labels=["is_above"]))


        """when"""
        fit = df.fit(model, test_size=0.4, test_validate_split_seed=42)
        fit_summary_df = fit.training_summary.df
        bt_summary_df = df.backtest(fit.model).df
        predict_df = df.predict(fit.model, tail=1)

        """then"""
        self.assertListEqual(fit_summary_df.columns.tolist(),
                             [(PREDICTION_COLUMN_NAME, 'is_above'), (LABEL_COLUMN_NAME, 'is_above')])
        self.assertEqual(len(fit_summary_df), 4022)

        self.assertListEqual(predict_df.columns.tolist(),
                             [(PREDICTION_COLUMN_NAME, 'is_above')])

        self.assertEqual(bt_summary_df.shape, (6704, 13))

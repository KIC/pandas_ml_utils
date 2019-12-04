import logging
import os
import unittest

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

import pandas_ml_utils as pdu
from pandas_ml_utils.model.features_and_labels.target_encoder import OneHotEncodedTargets
from pandas_ml_utils.constants import *
from test.utils import SMA

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test.csv")


class ClassificationTest(unittest.TestCase):

    def test_binary_classification(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["sma"] = SMA(df["spy_Close"])
        df["is_above"] = (df["spy_Close"] / df["sma"]) > 1

        model = pdu.SkitModel(
            MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
            pdu.FeaturesAndLabels(features=['vix_Close'],
                                  labels=["is_above"],
                                  targets=lambda _, frame: frame["sma"],
                                  loss=lambda _, frame: frame["spy_Close"] - frame["sma"]))

        """when"""
        fit = df.fit(model, test_size=0.4, test_validate_split_seed=42)
        fit_summary_df = fit.training_summary.df
        bt_summary_df = df.backtest(fit.model).df
        predict_df = df.predict(fit.model, tail=1)

        """then"""
        self.assertListEqual(fit_summary_df.columns.tolist(), [(PREDICTION_COLUMN_NAME, 'is_above'), (LABEL_COLUMN_NAME, 'is_above'), (LOSS_COLUMN_NAME, 'loss'), (TARGET_COLUMN_NAME, 'sma')])
        self.assertEqual(len(fit_summary_df), 4023)

        self.assertListEqual(bt_summary_df.columns.tolist(), fit_summary_df.columns.tolist())
        self.assertEqual(len(bt_summary_df), 6706)

        self.assertListEqual(predict_df.columns.tolist(), [(PREDICTION_COLUMN_NAME, 'is_above'), (TARGET_COLUMN_NAME, 'sma')])

    def test_multi_class_classification(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["sma"] = SMA(df["spy_Close"])
        df["label"] = df["spy_Close"] / df["sma"] -1

        def make_targets(t, frame):
            space = np.array([-1, -0.05, 0.5, 1])
            res = frame.apply(lambda x: x["sma"] - space, axis=1,
                              result_type='expand')
            res.columns = ["close <0.1", "close <0.05", "close >0", "close >0.05"]
            return res

        model = pdu.SkitModel(
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
                             [(PREDICTION_COLUMN_NAME, 'label #0'), (PREDICTION_COLUMN_NAME, 'label #1'), (PREDICTION_COLUMN_NAME, 'label #2'), (PREDICTION_COLUMN_NAME, 'label #3'),
                              (LABEL_COLUMN_NAME, 'label #0'), (LABEL_COLUMN_NAME, 'label #1'), (LABEL_COLUMN_NAME, 'label #2'), (LABEL_COLUMN_NAME, 'label #3'),
                              (TARGET_COLUMN_NAME, 'close <0.1'), (TARGET_COLUMN_NAME, 'close <0.05'), (TARGET_COLUMN_NAME, 'close >0'), (TARGET_COLUMN_NAME, 'close >0.05')])

        self.assertListEqual(bt_summary_df.columns.tolist(), fit_summary_df.columns.tolist())

        self.assertListEqual(predict_df.columns.tolist(),
                             [(PREDICTION_COLUMN_NAME, 'label #0'), (PREDICTION_COLUMN_NAME, 'label #1'), (PREDICTION_COLUMN_NAME, 'label #2'), (PREDICTION_COLUMN_NAME, 'label #3'),
                              (TARGET_COLUMN_NAME, 'close <0.1'), (TARGET_COLUMN_NAME, 'close <0.05'), (TARGET_COLUMN_NAME, 'close >0'), (TARGET_COLUMN_NAME, 'close >0.05')])

    def test_target_classification(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["sma"] = SMA(df["spy_Close"])
        df["is_above_1.0"] = (df["spy_Close"] / df["sma"]) > 1
        df["is_above_1.2"] = (df["spy_Close"] / df["sma"]) > 1.2

        model = pdu.SkitModel(
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

        self.assertListEqual(bt_summary_df.columns.tolist(), fit_summary_df.columns.tolist())

        self.assertListEqual(predict_df.columns.tolist(),
                             [('a', PREDICTION_COLUMN_NAME, 'is_above_1.0'), ('b', PREDICTION_COLUMN_NAME, 'is_above_1.2')])


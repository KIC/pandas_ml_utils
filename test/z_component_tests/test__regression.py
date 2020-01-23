import logging
import unittest

import pandas as pd
from sklearn.neural_network import MLPRegressor

import pandas_ml_utils as pdu
from test.config import TEST_FILE

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class RegressionTest(unittest.TestCase):

    def test_fit_regressor(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date') / 50.

        """when"""
        fit = df.fit(
            pdu.SkModel(
                MLPRegressor(activation='tanh', hidden_layer_sizes=(4, 3, 2, 1, 2, 3, 4), random_state=42),
                pdu.FeaturesAndLabels(
                    features=['spy_Open', 'spy_High', 'spy_Low', 'spy_Close'],
                    labels=['vix_Open', 'vix_High', 'vix_Low', 'vix_Close'],
                    targets=lambda frame: frame[['vix_Open', 'vix_High', 'vix_Low', 'vix_Close']].add_prefix("tgt_")
                )
            ),
            test_size=0.4,
            test_validate_split_seed=42)

        fitted_model = fit.model

        # backtest
        backtest_regression = df.backtest(fitted_model)
        self.assertIsNotNone(backtest_regression)

        # regressed
        regressed = df.predict(fitted_model)

        """then"""
        self.assertListEqual(regressed.columns.tolist(),
                             [('prediction', 'vix_Open'),
                              ('prediction', 'vix_High'),
                              ('prediction', 'vix_Low'),
                              ('prediction', 'vix_Close'),
                              ('target', 'tgt_vix_Open'),
                              ('target', 'tgt_vix_High'),
                              ('target', 'tgt_vix_Low'),
                              ('target', 'tgt_vix_Close')])

        self.assertEqual(len(regressed), 6706)

    def test_fit_regressor_mutiple_target(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date') / 50.

        """when"""
        fit = df.fit(
            pdu.SkModel(
                MLPRegressor(activation='tanh', hidden_layer_sizes=(4, 3, 2, 1, 2, 3, 4), random_state=42),
                pdu.FeaturesAndLabels(
                    features=['spy_Open', 'spy_High', 'spy_Low', 'spy_Close'],
                    labels={"a": ['vix_Open'], "b": ['vix_High', 'vix_Low', 'vix_Close']},
                    targets=lambda frame, t: frame[['vix_High', 'vix_Low']].add_prefix(f"{t}_")
                )
            ),
            test_size=0.4,
            test_validate_split_seed=42)

        fitted_model = fit.model

        # backtest
        backtest_regression = df.backtest(fitted_model)
        self.assertIsNotNone(backtest_regression)

        # regressed
        regressed = df.predict(fitted_model)

        """then"""
        self.assertListEqual(regressed.columns.tolist(),
                             [('a', 'prediction', 'vix_Open'),
                              ('b', 'prediction', 'vix_High'), ('b', 'prediction', 'vix_Low'), ('b', 'prediction', 'vix_Close'),
                              ('a', 'target', 'a_vix_High'), ('a', 'target', 'a_vix_Low'),
                              ('b', 'target', 'b_vix_High'), ('b', 'target', 'b_vix_Low')])

        self.assertEqual(len(regressed), 6706)

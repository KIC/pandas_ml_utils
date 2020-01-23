import logging
import unittest

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

import pandas_ml_utils as pdu
from test.config import TEST_FILE

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class HyportParameterOptimizationTest(unittest.TestCase):

    def test_hyper_parameter(self):
        from hyperopt import hp

        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df['label'] = df["spy_Close"] > df["spy_Open"]

        """when fit with find hyper parameter"""
        fit = df.fit(
            pdu.SkModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
                        pdu.FeaturesAndLabels(features=['vix_Close'], labels=['label'],
                                                target_columns=["vix_Open"],
                                                loss_column="spy_Volume")),
            test_size=0.4,
            test_validate_split_seed=42,
            hyper_parameter_space={'alpha': hp.choice('alpha', [0.0001, 10]), 'early_stopping': True, 'max_iter': 50,
                                   '__max_evals': 4, '__rstate': np.random.RandomState(42)}
        )

        """then test best parameter"""
        self.assertEqual(fit.model.skit_model.get_params()['alpha'], 0.0001)


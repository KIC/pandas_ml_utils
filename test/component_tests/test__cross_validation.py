import logging
import os
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

import pandas_ml_utils as pdu

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test.csv")


class CrossValidationTest(unittest.TestCase):

    def test_cross_validation(self):
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df['label'] = df["spy_Close"] > df["spy_Open"]

        # KFold
        cv = KFold(n_splits=10, shuffle=False)

        # fit
        fit = df.fit_classifier(pdu.SkitModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), alpha=0.001,
                                                            random_state=42, max_iter=10),
                                              pdu.FeaturesAndLabels(features=['vix_Close'],
                                                                    labels=['label'],
                                                                    targets=("vix_Open", "spy_Volume"))),
                                test_size=0.4,
                                cross_validation = (2, cv.split),
                                test_validate_split_seed=42)

        self.assertEqual(fit.model.features_and_labels.min_required_samples, 1)
        np.testing.assert_array_equal(fit.test_summary.get_confusion_matrix()["vix_Open"],
                                      np.array([[257, 169], [1142, 1115]]))


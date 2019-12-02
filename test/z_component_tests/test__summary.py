import logging
import os
import unittest

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

import pandas_ml_utils as pdu
from pandas_ml_utils.summary.binary_classification_summary import BinaryClassificationSummary
from pandas_ml_utils.utils.functions import fig_to_png_base64

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test.csv")


class ClassificationTest(unittest.TestCase):

    def test_binary_classification_summary(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df['label'] = df["spy_Close"] > df["spy_Open"]

        """when"""
        fit = df.fit(
            pdu.SkitModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), alpha=0.001, random_state=42),
                pdu.FeaturesAndLabels(features=['vix_Close'], labels=['label'],
                                      loss=lambda _, df: df["spy_Close"] - df["spy_Open"]),
                BinaryClassificationSummary),
            test_size=0.4,
            test_validate_split_seed=42)

        """then confusion matrix"""
        self.assertEqual(fit.model.features_and_labels.min_required_samples, 1)
        np.testing.assert_array_equal(fit.training_summary.get_confusion_matrix(), np.array([[1067,  872], [1002, 1082]]))
        np.testing.assert_array_equal(fit.test_summary.get_confusion_matrix(), np.array([[744, 586], [655, 698]]))

        """  and confusion loss"""
        np.testing.assert_array_almost_equal(fit.test_summary.get_confusion_loss(),
                                             np.array([[374.90, -234.83], [561.48, -650.63]]), 2)

        """  and ratios"""
        np.testing.assert_array_almost_equal(np.array(fit.test_summary.get_ratios()),
                                             np.array((0.78, 0.88)), 2)

        """  and metrics"""
        np.testing.assert_array_almost_equal(np.array(list(fit.test_summary.get_metrics().values())),
                                             np.array([0.78, 0.88, 0.54]), 2)

        """  and plot_classification"""
        self.assertDictEqual({k: repr(v) for k,v in fit.test_summary.plot_classification().items()},
                             {None: '<Figure size 1600x900 with 2 Axes>'})
        self.assertDictEqual({k: len(fig_to_png_base64(v)) for k,v in fit.test_summary.plot_classification().items()},
                             {None: 138908})

        """  and _repr_html_"""
        self.assertEqual(len(fit.test_summary._repr_html_()), 142212)
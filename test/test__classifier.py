import logging
from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ml_utils.classification.classifier import _convert_probabilities

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

target = [0.0, 0.0, 0.0, 0.0, 0.0]
loss = np.array([1.0, 1.0, 1.0, 1.0, 1.0]) * -1

df = pd.DataFrame({
    ("no fit", "target", "value"):          target,
    ("no fit", "prediction", "value"):      [1.0, 0.0, 1.0, 1.0, 0.0],
    ("no fit", "label", "value"):           [False, True, False, False, True],
    ("no fit", "loss", "value"):            loss
})


class TestClassifier(TestCase):

    def test__convert_probabilities(self):
        """given"""
        cut_off = 0.5

        """when"""
        augmented = _convert_probabilities(df, cut_off)

        """then"""
        self.assertTrue(("no fit", "prediction", "value_proba") in augmented.columns)
        self.assertEqual(augmented[("no fit", "prediction", "value_proba")].dtype, float)
        self.assertEqual(augmented[("no fit", "prediction", "value")].dtype, bool)

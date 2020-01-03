from unittest import TestCase

import numpy as np

from pandas_ml_utils.sampling.boosting import KFoldBoostRareEvents, KEquallyWeightEvents
from pandas_ml_utils.utils.classes import ReScaler
from pandas_ml_utils.utils.functions import unfold_parameter_space, one_hot


class TestUtilClasses(TestCase):

    def test_rescaler(self):
        """given"""
        data = np.random.random(50)
        scaler = ReScaler((data.min(), data.max()), (1, 2))

        """when"""
        scaled = scaler(data)

        """then"""
        np.testing.assert_almost_equal(scaled.min(), 1)
        np.testing.assert_almost_equal(scaled.max(), 2)
        self.assertEqual(np.argmax(data), np.argmax(scaled))

    def test_rescaler_reverse(self):
        """given"""
        data = np.random.random(50)
        scaler = ReScaler((data.min(), data.max()), (2, 1))

        """when"""
        scaled = scaler(data)

        """then"""
        np.testing.assert_almost_equal(scaled.min(), 1)
        np.testing.assert_almost_equal(scaled.max(), 2)
        self.assertNotEqual(np.argmax(scaled), len(data) - 1 - np.argmax(data))
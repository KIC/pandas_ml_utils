from unittest import TestCase

import numpy as np

from pandas_ml_utils.utils.functions import unfold_parameter_space
from pandas_ml_utils.utils.classes import ReScaler


class TestUtils(TestCase):

    def test_unfold_parameter_space(self):
        """given"""
        parameter_space = {"a": range(3, 6), "b": [1, 2, 3], "c": [4, 5]}

        """when"""
        unfolded_parameter_space = unfold_parameter_space(parameter_space, {})

        """then"""
        self.assertEqual(len(unfolded_parameter_space), 3 * 3 * 2)

    def test_rescaling(self):
        """given"""
        array = np.array([[1, 2], [3, 4]])
        rescaler = ReScaler((array.min(), array.max()), (-1, 1))

        """when"""
        resacled = rescaler(array)
        print(resacled)

        """then"""
        self.assertEqual(resacled[0, 0], -1)
        self.assertEqual(resacled[1, 1], 1)

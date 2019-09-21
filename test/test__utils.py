from unittest import TestCase

from pandas_ml_utils.utils import unfold_parameter_space


class TestUtils(TestCase):

    def test_unfold_parameter_space(self):
        parameter_space = {"a": range(3, 6), "b": [1, 2, 3], "c": [4, 5]}
        unfolded_parameter_space = unfold_parameter_space(parameter_space, {})
        self.assertEqual(len(unfolded_parameter_space), 3 * 3 * 2)

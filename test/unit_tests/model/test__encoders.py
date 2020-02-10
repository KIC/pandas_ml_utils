from unittest import TestCase
import pandas as pd
import numpy as np
from pandas._libs.interval import Interval

from pandas_ml_utils.model.features_and_labels.target_encoder import OneHotEncodedDiscrete, OneHotEncodedTargets


class TestEncoders(TestCase):

    def test__one_hot_encoded_targets(self):
        """given"""
        df = pd.DataFrame({"a": [-0.1, 0, 0.1], "b": [0, 1, 2]})
        encoder = OneHotEncodedTargets("a", np.linspace(-0.1, 0.1, 4, endpoint=True))

        """when"""
        encoded = encoder.encode(df)
        decoded = encoder.decode(encoded)

        """then"""
        np.testing.assert_array_almost_equal(encoded.values, np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ]))
        self.assertEqual(decoded[1], Interval(-0.03333333333333334, 0.033333333333333326, closed='right'))

    def test__one_hot_discrete(self):
        """given"""
        df = pd.DataFrame({"a": [0, 1, 2], "b": [0, 1, 2]})
        encoder = OneHotEncodedDiscrete("a", 3)

        """when"""
        encoded = encoder.encode(df)
        decoded = encoder.decode(encoded)

        """then"""
        np.testing.assert_array_almost_equal(encoded.values, np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ]))
        np.testing.assert_array_equal(decoded, np.array([0, 1, 2]))

    def test__one_hot_discrete_with_preprocessor(self):
        """given"""
        df = pd.DataFrame({"a": [0, 1, 2], "b": [0, 1, 2]})
        encoder = OneHotEncodedDiscrete("a", 5, pre_processor=lambda x, fact: x * fact)

        """when"""
        encoded = encoder.encode(df, fact=2, foo=12)
        decoded = encoder.decode(encoded)

        """then"""
        np.testing.assert_array_almost_equal(encoded.values, np.array([
            [1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1.]
        ]))
        np.testing.assert_array_equal(decoded, np.array([0, 2, 4]))
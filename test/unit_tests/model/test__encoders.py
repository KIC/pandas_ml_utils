from unittest import TestCase
import pandas as pd
import numpy as np
from pandas_ml_utils.model.features_and_labels.target_encoder import OneHotEncodedDiscrete


class TestEncoders(TestCase):

    def test__one_hot_discrete(self):
        """given"""
        df = pd.DataFrame({"a": [0, 1, 2], "b": [0, 1, 2]})
        encoder = OneHotEncodedDiscrete("a", 3)

        """when"""
        encoded = encoder.encode(df)

        """then"""
        np.testing.assert_array_almost_equal(encoded.values, np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ]))
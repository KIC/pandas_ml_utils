from unittest import TestCase

from pandas_ml_utils.model.features_and_Labels import FeaturesAndLabels


class TestFeaturesAndLabels(TestCase):

    def test_1d(self):
        """given"""
        fl = FeaturesAndLabels(["a", "b", "c"], ["d", "e"])

        """when"""
        shape = fl.shape()

        """then"""
        self.assertEqual(shape, ((3, ), (2, )))


    def test_2d(self):
        """given"""
        fl = FeaturesAndLabels(["a", "b", "c"], ["d", "e"], feature_lags=range(4))

        """when"""
        shape = fl.shape()

        """then"""
        # shape is ((timesteps, features), (labels, )
        self.assertEqual(shape, ((4, 3), (2, )))
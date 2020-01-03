from unittest import TestCase

from pandas_ml_utils.model.features_and_labels.features_and_labels import FeaturesAndLabels
from test.utils import SMA


class TestFeaturesAndLabels(TestCase):

    def test_id(self):
        """given"""
        fl1 = FeaturesAndLabels(["a", "b", "c"], ["d", "e"], targets={"b": None})
        fl2 = FeaturesAndLabels(["a", "b", "c"], ["d", "e"], targets={"b": None})
        fl3 = FeaturesAndLabels(["a", "b", "d"], ["d", "e"], targets={"b": None})

        """expect"""
        self.assertEqual(fl1.__id__(), fl2.__id__())
        self.assertEqual(hash(fl2), hash(fl2))
        self.assertNotEqual(hash(fl2), hash(fl3))

    def test_1d(self):
        """given"""
        fl = FeaturesAndLabels(["a", "b", "c"], ["d", "e"])

        """when"""
        shape = fl.shape

        """then"""
        self.assertEqual(shape, ((3, ), (2, )))


    def test_2d(self):
        """given"""
        fl = FeaturesAndLabels(["a", "b", "c"], ["d", "e"], feature_lags=range(4))

        """when"""
        shape = fl.shape

        """then"""
        # shape is ((timesteps, features), (labels, )
        self.assertEqual(shape, ((4, 3), (2, )))

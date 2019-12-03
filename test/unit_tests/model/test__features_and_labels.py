from unittest import TestCase

from pandas_ml_utils.model.features_and_labels.features_and_labels import FeaturesAndLabels
import talib


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

    def test_min_required_samples(self):
        """when"""
        fl1 = FeaturesAndLabels(["a", "b", "c"], ["d", "e"],
                                feature_lags=[1])
        fl2 = FeaturesAndLabels(["a", "b", "c"], ["d", "e"],
                                feature_lags=[1],
                                lag_smoothing={1: lambda df: talib.SMA(df[df.columns[0]], timeperiod=2)})

        """then"""
        # original | lagged | smoothed
        # 1        |        |
        # 2        | 1      |
        self.assertEqual(fl1.min_required_samples, 1 + 1)

        # original | lagged | smoothed
        # 1        |        |
        # 2        | 1      |
        # 3        | 2      | 1.5
        self.assertEqual(fl2.min_required_samples, 1 + 1 + (2 -1))
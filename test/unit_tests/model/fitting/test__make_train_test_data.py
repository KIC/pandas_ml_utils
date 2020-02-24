import unittest

import numpy as np
import pandas as pd

import pandas_ml_utils as pdu
from pandas_ml_utils.model.features_and_labels.features_and_labels_extractor import FeatureTargetLabelExtractor
from pandas_ml_utils.model.fitting.splitting import train_test_split


class TestTrainTestData(unittest.TestCase):

    def test_no_training_data(self):
        """given"""
        df = pd.DataFrame({"featureA": [1,2,3,4,5],
                           "labelA": [1,2,3,4,5]})

        """when"""
        train_ix, test_ix = train_test_split(df.index, 0)

        """then"""
        np.testing.assert_array_almost_equal(train_ix.values, df.index.values)
        self.assertEqual(0, len(test_ix))

    def test_make_training_data(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5],
                           "labelA": [1, 2, 3, 4, 5]})

        """when"""
        train_ix, test_ix = train_test_split(df.index, test_size=0.5)

        """then"""
        self.assertEqual(2, len(train_ix))
        self.assertEqual(3, len(test_ix))

    def test_make_youngest_training_data(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5],
                           "labelA": [1, 2, 3, 4, 5]})

        """when"""
        normal_train, normal_test = train_test_split(df.index, test_size=0.5, seed='youngest')
        lagged_train, lagged_test = train_test_split(
            FeatureTargetLabelExtractor(df, pdu.FeaturesAndLabels(["featureA"], ["labelA"], feature_lags=[0, 1])).features_labels_weights_df[0].index,
            test_size=0.5,
            seed='youngest')

        """then expect a test set"""
        self.assertEqual(3, len(normal_test))
        np.testing.assert_array_equal(normal_test[-3:], np.array([2, 3, 4]))
        self.assertEqual(2, len(lagged_train))
        np.testing.assert_array_equal(lagged_test[-2:], np.array([3, 4]))

    def test_youngest_portion(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "labelA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        """when"""
        train_ix, test_ix = train_test_split(df.index, test_size=0.6, youngest_size=0.25)

        "then"
        self.assertEqual(6, len(test_ix))
        np.testing.assert_array_equal(test_ix[-2:], np.array([8, 9]))

    def test_lag_smoothing_nan(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "labelA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "labelB": [5, 4, 3, 2, 1, 0, 1, 2, 3, None]})

        """when lag smoothing is enabled using shift (which is introducing nan into the data frame)"""
        fl = pdu.FeaturesAndLabels(["featureA"], ["labelB"], feature_lags=[0, 1],
                                   lag_smoothing={1: lambda df: df["featureA"].shift(2)})

        f, l, _ = FeatureTargetLabelExtractor(df, fl).features_labels_weights_df
        len_features = 10 - 1 - 2
        len_none_lables = 1

        """then"""
        self.assertEqual(len(f), len_features - len_none_lables)
        self.assertAlmostEqual(f["featureA", 1].iloc[0], 1.0)
        self.assertAlmostEqual(f["featureA", 1].iloc[-1], 6.0)

    def test_hashable_features_and_labels(self):
        """given"""
        a = pdu.FeaturesAndLabels(["featureA"], ["featureA"], feature_lags=[1, 2, 3, 4],
                                  lag_smoothing={2: lambda df: df[["featureA"]] * 2,
                                                 4: lambda df: df[["featureA"]] * 4})

        b = pdu.FeaturesAndLabels(["featureA"], ["featureA"], feature_lags=[1, 2, 3, 4],
                                  lag_smoothing={2: lambda df: df[["featureA"]] * 2,
                                                 4: lambda df: df[["featureA"]] * 4})

        """expect"""
        self.assertEqual(hash(a), hash(a))
        self.assertEqual(a, a)

        self.assertEqual(a.__id__(), b.__id__())
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_feature_scaling_3d(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "featureC": [11, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "featureB": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           "labelA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        """when"""
        fl = pdu.FeaturesAndLabels(["featureA", "featureB", "featureC"],
                                   ["labelA"],
                                   feature_lags=[1, 2],
                                   feature_rescaling={("featureA", "featureC"): (-1, 1)})

        f = FeatureTargetLabelExtractor(df, fl).features_df.values

        """then"""
        self.assertEqual((8, 2, 3), f.shape)
        self.assertEqual(f[0, 1, 2], 1)
        self.assertEqual(f[0, 1, 0], -1)
        np.testing.assert_array_almost_equal(f[0, :, 1], df["featureB"][[1, 0]])

    def test_feature_scaling_2d(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "featureC": [11, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "featureB": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           "labelA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        """when"""
        fl = pdu.FeaturesAndLabels(["featureA", "featureB", "featureC"],
                                   ["labelA"],
                                   feature_rescaling={("featureA", "featureC"): (-1, 1)})

        f = FeatureTargetLabelExtractor(df, fl).features_df.values

        "then"
        np.testing.assert_array_almost_equal(f[0], np.array([-1, 0.1, 1]))

    def test_lagging(self):
        """given"""
        df = pd.DataFrame({"featureA": [0.5592344 , 0.60739384, 0.19994533, 0.56642537, 0.50965677,
                                        0.168989  , 0.94080671, 0.76651769, 0.8403563 , 0.4003567 ,
                                        0.24295908, 0.50706317, 0.66612371, 0.4020924 , 0.21776017,
                                        0.32559497, 0.12721287, 0.13904584, 0.65887554, 0.08830925],
                           "featureC": [0.43700002, 0.36804634, 0.37568437, 0.34575936, 0.45747071,
                                        0.49749949, 0.09991126, 0.64710179, 0.93479635, 0.40651901,
                                        0.22387503, 0.18458239, 0.0271485 , 0.68792433, 0.69638729,
                                        0.60715125, 0.52557556, 0.88319929, 0.34808869, 0.50250121],
                           "featureB": [0.6784767 , 0.184668  , 0.02049038, 0.92983967, 0.67628553,
                                        0.71373065, 0.30594832, 0.63038278, 0.78284284, 0.84566334,
                                        0.00558188, 0.15819783, 0.09404578, 0.34460875, 0.69161826,
                                        0.41633249, 0.51130681, 0.66703763, 0.74652599, 0.26560367],
                           "labelA": range(20)})

        """when"""
        fl = pdu.FeaturesAndLabels(["featureA", "featureB", "featureC"],
                                   ["labelA"],
                                   feature_lags=[0,1,2,3,4])

        f = FeatureTargetLabelExtractor(df, fl).features_df.values

        """then"""
        self.assertEqual(len(f), len(df) - 4)
        np.testing.assert_array_equal(f[0,:,0], df["featureA"].values[[4,3,2,1,0]])
        np.testing.assert_array_equal(f[-1,:,0], df["featureA"].values[[-1, -2, -3, -4, -5]])
        np.testing.assert_array_equal(f[0,:,1], df["featureB"].values[[4,3,2,1,0]])
        np.testing.assert_array_equal(f[-1,:,1], df["featureB"].values[[-1, -2, -3, -4, -5]])
        np.testing.assert_array_equal(f[0,:,2], df["featureC"].values[[4,3,2,1,0]])
        np.testing.assert_array_equal(f[-1,:,2], df["featureC"].values[[-1, -2, -3, -4, -5]])

    def test_make_prediction_data(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "labelA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        """when"""
        fl = pdu.FeaturesAndLabels(["featureA"], ["labelA"])
        f = FeatureTargetLabelExtractor(df, fl).features_df.values

        """then"""
        self.assertEqual((10, 1), f.shape)


if __name__ == '__main__':
    unittest.main()

import unittest
import pandas_ml_utils as pdu
import numpy as np
import pandas as pd


class TestTrainTestData(unittest.TestCase):

    def test_no_training_data(self):

        df = pd.DataFrame({"featureA": [1,2,3,4,5],
                           "featureB": [5,4,3,2,1],
                           "labelA": [1,2,3,4,5],
                           "labelB": [5,4,3,2,1]})

        x_train, x_test, y_train, y_test, _, _, _ = df.make_training_data(
            pdu.FeaturesAndLabels(["featureA", "featureB"], ["labelA"]),
            test_size=0)

        self.assertIsNone(x_test)
        self.assertIsNone(y_test)
        np.testing.assert_array_almost_equal(x_train, df[["featureA", "featureB"]].values)
        np.testing.assert_array_almost_equal(y_train, df["labelA"].values)

    def test_make_training_data(self):
        df = pd.DataFrame({"featureA": [1,2,3,4,5],
                           "featureB": [5,4,3,2,1],
                           "labelA": [1,2,3,4,5],
                           "labelB": [5,4,3,2,1]})

        x_train, x_test, y_train, y_test, _, _, _ = df.make_training_data(
            pdu.FeaturesAndLabels(["featureA", "featureB"], ["labelA"]),
            test_size=0.5)

        np.testing.assert_array_almost_equal(x_test, np.array([[2, 4], [5, 1], [3, 3]]))
        np.testing.assert_array_almost_equal(y_test, np.array([2, 5, 3]))

    def test_make_training_data_two_labels(self):
        df = pd.DataFrame({"featureA": [1,2,3,4,5],
                           "featureB": [5,4,3,2,1],
                           "labelA": [1,2,3,4,5],
                           "labelB": [5,4,3,2,1]})

        x_train, x_test, y_train, y_test, _, _, _  = df.make_training_data(
            pdu.FeaturesAndLabels(["featureA", "featureB"], ["labelA", "labelB"]),
            test_size=0.5)

        np.testing.assert_array_almost_equal(x_test, np.array([[2, 4], [5, 1], [3, 3]]))
        np.testing.assert_array_almost_equal(y_test, np.array([[2, 4], [5, 1], [3, 3]]))

    def test_make_rnn_training_data(self):
        df = pd.DataFrame({"featureA": [1,2,3,4,5],
                           "featureB": [5,4,3,2,1],
                           "labelA": [1,2,3,4,5],
                           "labelB": [5,4,3,2,1]})

        fl = pdu.FeaturesAndLabels(["featureA", "featureB"], ["labelA"], feature_lags=[0, 1])

        x_train, x_test, y_train, y_test, _, _, min = df.make_training_data(fl, test_size=0.5)

        # test whole shape and labels
        np.testing.assert_array_almost_equal(x_test, np.array([[[3, 3], [2, 4]], [[5, 1], [4, 2]]]))
        np.testing.assert_array_almost_equal(y_test, np.array([3, 5]))

        # all rows, all lags one feature -> feature[0] needs lag of -1
        np.testing.assert_array_almost_equal(x_test[:,:,0], np.array([[3, 2], [5, 4]]))
        # all rows, all lags one feature -> feature[1] needs lag of +1
        np.testing.assert_array_almost_equal(x_test[:,:,1], np.array([[3, 4], [1, 2]]))

        # test min
        self.assertEqual(min, 2)

        # test names
        self.assertListEqual(fl.get_feature_names().tolist(), [['featureA_0', 'featureB_0'],
                                                               ['featureA_1', 'featureB_1']])

    def test_make_single_lagged_training_data(self):
        df = pd.DataFrame({"featureA": [1,2,3,4,5],
                           "featureB": [5,4,3,2,1],
                           "labelA": [1,2,3,4,5],
                           "labelB": [5,4,3,2,1]})

        x_train, x_test, y_train, y_test, _, _, _ = df.make_training_data(
            pdu.FeaturesAndLabels(["featureA"], ["labelA"], feature_lags=[1, 2]),
            test_size=0.5)

        np.testing.assert_array_almost_equal(x_test, np.array([[[2], [1]], [[3], [2]]]))
        np.testing.assert_array_almost_equal(y_test, np.array([3, 4]))

    def test_make_smoothed_training_data(self):
        df = pd.DataFrame({"featureA": [1,2,3,4,5,6,7,8,9,10],
                           "featureB": [5,4,3,2,1,0,1,2,3,4],
                           "labelA": [1,2,3,4,5,6,7,8,9,10],
                           "labelB": [5,4,3,2,1,0,1,2,3,4]})

        x_train, x_test, y_train, y_test, _, _, min = df.make_training_data(
            pdu.FeaturesAndLabels(["featureA"], ["featureA"], feature_lags=[1, 2, 3, 4],
                                  lag_smoothing={2: lambda df: df[["featureA"]] * 2,
                                                 4: lambda df: df[["featureA"]] * 4}),
            test_size=0.5)

        np.testing.assert_array_almost_equal(x_train[-1], [[7], [12], [10], [16]])
        self.assertEqual(min, 5)

    def test_lag_smoothing_nan(self):
        # test lag smoothing using shift (introducing nan)
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "labelA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "labelB": [5, 4, 3, 2, 1, 0, 1, 2, 3, None]})

        from pandas_ml_utils.train_test_data import _make_features
        fl = pdu.FeaturesAndLabels(["featureA"], None, feature_lags=[0, 1],
                                   lag_smoothing={1: lambda df: df["featureA"].shift(2)})

        df, x = _make_features(df, fl)
        len_features = 10 - 1 - 2
        len_none_lables = 1

        self.assertEqual(len(df), len_features - len_none_lables)
        np.testing.assert_array_equal(fl.get_feature_names(), np.array( [['featureA_0'], ['featureA_1']]))
        self.assertAlmostEqual(df["featureA_1"].iloc[0], 1.0)
        self.assertAlmostEqual(df["featureA_1"].iloc[-1], 6.0)

    def test_reshape_rnn_as_ar(self):
        np.testing.assert_array_almost_equal(pdu.reshape_rnn_as_ar(np.array([[[1], [2]]], ndmin=3)),
                                             np.array([[1, 2]], ndmin=2))

        np.testing.assert_array_almost_equal(pdu.reshape_rnn_as_ar(np.array([[1, 2]], ndmin=2)),
                                             np.array([[1, 2]], ndmin=2))

    def test_hashable_features_and_labels(self):
        a = pdu.FeaturesAndLabels(["featureA"], ["featureA"], feature_lags=[1, 2, 3, 4],
                                  lag_smoothing={2: lambda df: df[["featureA"]] * 2,
                                                 4: lambda df: df[["featureA"]] * 4})

        b = pdu.FeaturesAndLabels(["featureA"], ["featureA"], feature_lags=[1, 2, 3, 4],
                                  lag_smoothing={2: lambda df: df[["featureA"]] * 2,
                                                 4: lambda df: df[["featureA"]] * 4})

        self.assertEqual(hash(a), hash(a))
        self.assertEqual(a, a)

        self.assertEqual(a.__id__(), b.__id__())
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_cache_feature_matrix(self):
        from pandas_ml_utils.train_test_data import make_training_data, _make_features_with_cache

        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "labelA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "labelB": [5, 4, 3, 2, 1, 0, 1, 2, 3, None]})

        ful = pdu.FeaturesAndLabels(["featureA"], ["featureA"], feature_lags=[1, 2, 3, 4],
                                    lag_smoothing={2: lambda df: df[["featureA"]] * 2,
                                                   4: lambda df: df[["featureA"]] * 4})

        x_train, x_test, y_train, y_test, index_train, index_test, min = make_training_data(df, ful, cache=True)
        x_train, x_test, y_train, y_test, index_train, index_test, min = make_training_data(df, ful, cache=True)

        cache_info = _make_features_with_cache.cache_info()
        print(cache_info)

        self.assertEqual(cache_info.misses, 1)  # miss the first call
        self.assertEqual(cache_info.hits, 1)    # fetch the first cached call

    def test_feature_scaling_3d(self):
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "featureC": [11, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "featureB": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           "labelA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        fl = pdu.FeaturesAndLabels(["featureA", "featureB", "featureC"],
                                   ["labelA"],
                                   feature_lags=[1, 2],
                                   feature_rescaling={("featureA", "featureC"): (-1, 1)})

        x_train, x_test, y_train, y_test, _, _, _ = df.make_training_data(fl, test_size=0)

        print(x_train)
        self.assertEqual(x_train.argmax(), 5)
        self.assertEqual(x_train[0,1,2], 1)
        self.assertEqual(x_train[0,1,0], -1)
        np.testing.assert_array_almost_equal(x_train[0,:,1], df["featureB"][[1,0]])

    def test_feature_scaling_2d(self):
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "featureC": [11, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "featureB": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           "labelA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        fl = pdu.FeaturesAndLabels(["featureA", "featureB", "featureC"],
                                   ["labelA"],
                                   feature_rescaling={("featureA", "featureC"): (-1, 1)})

        x_train, x_test, y_train, y_test, _, _, _ = df.make_training_data(fl, test_size=0)

        print(x_train)
        np.testing.assert_array_almost_equal(x_train[0], np.array([-1, 0.1, 1]))


if __name__ == '__main__':
    unittest.main()

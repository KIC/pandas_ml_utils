import unittest
import pd_utils as pdu
import numpy as np
import pandas as pd


class TestTrainTestData(unittest.TestCase):

    def test_no_training_data(self):

        df = pd.DataFrame({"featureA": [1,2,3,4,5],
                           "featureB": [5,4,3,2,1],
                           "labelA": [1,2,3,4,5],
                           "labelB": [5,4,3,2,1]})

        x_train, x_test, y_train, y_test, _, _, _ = df.make_training_data(pdu.FeaturesAndLabels(["featureA", "featureB"],
                                                                                                ["labelA"]),
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

        x_train, x_test, y_train, y_test, _, _, _ = df.make_training_data(pdu.FeaturesAndLabels(["featureA", "featureB"],
                                                                                                ["labelA"]),
                                                                          test_size=0.5)

        np.testing.assert_array_almost_equal(x_test, np.array([[2, 4], [5, 1], [3, 3]]))
        np.testing.assert_array_almost_equal(y_test, np.array([2, 5, 3]))

    def test_make_training_data_two_labels(self):
        df = pd.DataFrame({"featureA": [1,2,3,4,5],
                           "featureB": [5,4,3,2,1],
                           "labelA": [1,2,3,4,5],
                           "labelB": [5,4,3,2,1]})

        x_train, x_test, y_train, y_test, _, _, _ = df.make_training_data(pdu.FeaturesAndLabels(["featureA", "featureB"],
                                                                                                ["labelA", "labelB"]),
                                                                          test_size=0.5)

        np.testing.assert_array_almost_equal(x_test, np.array([[2, 4], [5, 1], [3, 3]]))
        np.testing.assert_array_almost_equal(y_test, np.array([[2, 4], [5, 1], [3, 3]]))

    def test_make_rnn_training_data(self):
        df = pd.DataFrame({"featureA": [1,2,3,4,5],
                           "featureB": [5,4,3,2,1],
                           "labelA": [1,2,3,4,5],
                           "labelB": [5,4,3,2,1]})

        x_train, x_test, y_train, y_test, _, _, names = df.make_training_data(pdu.FeaturesAndLabels(["featureA", "featureB"],
                                                                                                    ["labelA"],
                                                                                                    feature_lags=[0, 1]),
                                                                              test_size=0.5)

        # test whole shape and labels
        np.testing.assert_array_almost_equal(x_test, np.array([[[3, 3], [2, 4]], [[5, 1], [4, 2]]]))
        np.testing.assert_array_almost_equal(y_test, np.array([3, 5]))

        # all rows, all lags one feature -> feature[0] needs lag of -1
        np.testing.assert_array_almost_equal(x_test[:,:,0], np.array([[3, 2], [5, 4]]))
        # all rows, all lags one feature -> feature[1] needs lag of +1
        np.testing.assert_array_almost_equal(x_test[:,:,1], np.array([[3, 4], [1, 2]]))

        # test names
        self.assertListEqual(names[0].tolist(), [['featureA_0', 'featureB_0'],
                                                 ['featureA_1', 'featureB_1']])

    def test_make_single_lagged_training_data(self):
        df = pd.DataFrame({"featureA": [1,2,3,4,5],
                           "featureB": [5,4,3,2,1],
                           "labelA": [1,2,3,4,5],
                           "labelB": [5,4,3,2,1]})

        x_train, x_test, y_train, y_test, _, _, _ = df.make_training_data(pdu.FeaturesAndLabels(["featureA"],
                                                                                                ["labelA"],
                                                                                                feature_lags=[1, 2]),
                                                                          test_size=0.5)

        np.testing.assert_array_almost_equal(x_test, np.array([[[2], [1]], [[3], [2]]]))
        np.testing.assert_array_almost_equal(y_test, np.array([3, 4]))

    def test_make_smoothed_training_data(self):
        df = pd.DataFrame({"featureA": [1,2,3,4,5,6,7,8,9,10],
                           "featureB": [5,4,3,2,1,0,1,2,3,4],
                           "labelA": [1,2,3,4,5,6,7,8,9,10],
                           "labelB": [5,4,3,2,1,0,1,2,3,4]})

        x_train, x_test, y_train, y_test, _, _, _ = df.make_training_data(pdu.FeaturesAndLabels(["featureA"],
                                                                                                ["featureA"],
                                                                                                feature_lags=[1, 2, 3, 4],
                                                                                                lag_smoothing={2: lambda df: df[["featureA"]] * 2,
                                                                                                               4: lambda df: df[["featureA"]] * 4}),
                                                                          test_size=0.5)

        np.testing.assert_array_almost_equal(x_train[-1], [[7], [12], [10], [16]])

    def test_lag_smoothing_nan(self):
        # test lag smoothing using shift (introducing nan)
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "labelA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "labelB": [5, 4, 3, 2, 1, 0, 1, 2, 3, None]})

        from pd_utils.make_train_test_data import _make_features
        df, x, names = _make_features(df, ["featureA"], [0, 1], {1: lambda df: df["featureA"].shift(2)})

        len_features = 10 - 1 - 2
        len_none_lables = 1

        self.assertEqual(len(df), len_features - len_none_lables)
        np.testing.assert_array_equal(names, np.array( [['featureA_0'], ['featureA_1']]))
        self.assertAlmostEqual(df["featureA_1"].iloc[0], 1.0)
        self.assertAlmostEqual(df["featureA_1"].iloc[-1], 6.0)

    def test_reshape_rnn_as_ar(self):
        np.testing.assert_array_almost_equal(pdu.reshape_rnn_as_ar(np.array([[[1], [2]]], ndmin=3)),
                                             np.array([[1, 2]], ndmin=2))

        np.testing.assert_array_almost_equal(pdu.reshape_rnn_as_ar(np.array([[1, 2]], ndmin=2)),
                                             np.array([[1, 2]], ndmin=2))


if __name__ == '__main__':
    unittest.main()
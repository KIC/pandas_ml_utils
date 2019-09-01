import unittest
import pd_utils
import numpy as np
import pandas as pd


class TestUtils(unittest.TestCase):

    def test_no_training_data(self):
        df = pd.DataFrame({"featureA": [1,2,3,4,5],
                           "featureB": [5,4,3,2,1],
                           "labelA": [1,2,3,4,5],
                           "labelB": [5,4,3,2,1]})

        x_train, x_test, y_train, y_test, _ = df.make_training_data(features=["featureA", "featureB"],
                                                                    labels=["labelA"],
                                                                    test_size=0,
                                                                    feature_lags=None).values()

        self.assertIsNone(x_test)
        self.assertIsNone(y_test)
        np.testing.assert_array_almost_equal(x_train, df[["featureA", "featureB"]].values)
        np.testing.assert_array_almost_equal(y_train, df["labelA"].values)

    def test_make_training_data(self):
        df = pd.DataFrame({"featureA": [1,2,3,4,5],
                           "featureB": [5,4,3,2,1],
                           "labelA": [1,2,3,4,5],
                           "labelB": [5,4,3,2,1]})

        x_train, x_test, y_train, y_test, _ = df.make_training_data(features=["featureA", "featureB"],
                                                                    labels=["labelA"],
                                                                    test_size=0.5,
                                                                    feature_lags=None).values()

        np.testing.assert_array_almost_equal(x_test, np.array([[2, 4], [5, 1], [3, 3]]))
        np.testing.assert_array_almost_equal(y_test, np.array([2, 5, 3]))

    def test_make_training_data_two_labels(self):
        df = pd.DataFrame({"featureA": [1,2,3,4,5],
                           "featureB": [5,4,3,2,1],
                           "labelA": [1,2,3,4,5],
                           "labelB": [5,4,3,2,1]})

        x_train, x_test, y_train, y_test, _ = df.make_training_data(features=["featureA", "featureB"],
                                                                    labels=["labelA", "labelB"],
                                                                    test_size=0.5,
                                                                    feature_lags=None).values()

        np.testing.assert_array_almost_equal(x_test, np.array([[2, 4], [5, 1], [3, 3]]))
        np.testing.assert_array_almost_equal(y_test, np.array([[2, 4], [5, 1], [3, 3]]))

    def test_make_rnn_training_data(self):
        df = pd.DataFrame({"featureA": [1,2,3,4,5],
                           "featureB": [5,4,3,2,1],
                           "labelA": [1,2,3,4,5],
                           "labelB": [5,4,3,2,1]})

        x_train, x_test, y_train, y_test, names = df.make_training_data(features=["featureA", "featureB"],
                                                                        labels=["labelA"],
                                                                        test_size=0.5,
                                                                        feature_lags=[0, 1]).values()

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

        x_train, x_test, y_train, y_test, _ = df.make_training_data(features=["featureA"],
                                                                    labels=["featureA"],
                                                                    test_size=0.5,
                                                                    feature_lags=[1, 2]).values()

        np.testing.assert_array_almost_equal(x_test, np.array([[[2], [1]], [[3], [2]]]))
        np.testing.assert_array_almost_equal(y_test, np.array([3, 4]))

    def test_make_smoothed_training_data(self):
        df = pd.DataFrame({"featureA": [1,2,3,4,5,6,7,8,9,10],
                           "featureB": [5,4,3,2,1,0,1,2,3,4],
                           "labelA": [1,2,3,4,5,6,7,8,9,10],
                           "labelB": [5,4,3,2,1,0,1,2,3,4]})

        x_train, x_test, y_train, y_test, _ = df.make_training_data(features=["featureA"],
                                                                    labels=["featureA"],
                                                                    lag_smoothing={2: lambda df: df[["featureA"]] * 2,
                                                                                   4: lambda df: df[["featureA"]] * 4},
                                                                    test_size=0.5,
                                                                    feature_lags=[1, 2, 3, 4]).values()


        np.testing.assert_array_almost_equal(x_train[-1], [[7], [12], [10], [16]])


if __name__ == '__main__':
    unittest.main()

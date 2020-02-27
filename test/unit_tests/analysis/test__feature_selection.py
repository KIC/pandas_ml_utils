import unittest

from pandas_ml_utils import pd


class TestFeatureSelection(unittest.TestCase):

    def test_feature_selection(self):
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5],
                           "featureB": [5, 4, 3, 2, 1],
                           "featureC": [1, 2, 1, 2, 1],
                           "labelA": [1, 2, 3, 4, 5],
                           "labelB": [5, 4, 3, 2, 1]})

        df.feature_selection("labelA", lags=[2], show_plots=False)


if __name__ == '__main__':
    unittest.main()

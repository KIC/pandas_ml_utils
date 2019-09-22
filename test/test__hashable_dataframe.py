import unittest
import pandas as pd
import math

from pandas_ml_utils.wrappers.hashable_dataframe import HashableDataFrame


class TestHashableDataFrame(unittest.TestCase):

    def test_hashable_dataframe(self):
        df = pd.DataFrame({"test": [1,2,3,4,5]})
        df["pi"] = df["test"] * math.pi

        self.assertEqual(hash(HashableDataFrame(df)), hash(HashableDataFrame(df)))
        self.assertEqual(HashableDataFrame(df), HashableDataFrame(df))


if __name__ == '__main__':
    unittest.main()

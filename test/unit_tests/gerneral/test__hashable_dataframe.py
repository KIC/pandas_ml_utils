import math
import unittest

import pandas_ml_utils.monkey_patched_dataframe as pd

from pandas_ml_utils.wrappers.hashable_dataframe import HashableDataFrame


class TestHashableDataFrame(unittest.TestCase):

    def test_hashable_dataframe(self):
        df = pd.DataFrame({"test": [1,2,3,4,5]})
        df["pi"] = df["test"] * math.pi

        self.assertEqual(hash(HashableDataFrame(df)), hash(HashableDataFrame(df)))
        self.assertEqual(HashableDataFrame(df), HashableDataFrame(df))


if __name__ == '__main__':
    unittest.main()

import unittest

import pandas as pd

from pandas_ml_utils.wrappers.lazy_dataframe import LazyDataFrame


class TestLazyDataFrame(unittest.TestCase):

    def test_lazy_dataframe(self):
        df = pd.DataFrame({"test": [1,2,3,4,5]})
        ldf = LazyDataFrame(df, foo=lambda df: df["test"] * 2)

        # test lazy functions
        pd.testing.assert_series_equal(ldf["foo"], df["test"] * 2)
        self.assertEqual(ldf[["foo", "test"]].shape, (5,2))
        pd.testing.assert_frame_equal(ldf[["test", "foo"]], ldf.to_dataframe())

        # test delegation
        pd.testing.assert_index_equal(ldf.index, df.index)
        self.assertEqual(len(ldf.min().index), 2)

        # test structure
        self.assertFalse("foo" in ldf.df)
        self.assertTrue("foo" in ldf)


if __name__ == '__main__':
    unittest.main()

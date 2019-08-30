import unittest
import pandas as pd
from pd_utils.lazy_dataframe import LazyDataFrame

class TestLazyDataFrame(unittest.TestCase):

    def test_lazy_dataframe(self):
        df = pd.DataFrame({"test": [1,2,3,4,5]})
        ldf = LazyDataFrame(df, foo=lambda df: df["test"] * 2)

        # test lazy functions
        pd.testing.assert_series_equal(ldf["foo"], df["test"] * 2)

        # test delegation
        pd.testing.assert_index_equal(ldf.index, df.index)
        pd.testing.assert_series_equal(ldf.min(), df.min())

        # test structure
        self.assertFalse("foo" in ldf.df)
        self.assertTrue("foo" in ldf)


if __name__ == '__main__':
    unittest.main()

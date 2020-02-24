from unittest import TestCase

import numpy as np
import pandas as pd

import pandas_ml_utils as pml

print(pml.__version__)


class TestDfExtensions(TestCase):

    def test__inner_join(self):
        """given"""
        dfa = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
        dfb = pd.DataFrame({"a": [1, 2, 3]}, index=[1, 2, 3])

        """when"""
        df = dfa.inner_join(dfb, prefix='b_')

        """then"""
        pd.testing.assert_frame_equal(df, pd.DataFrame({"a": [2, 3], "b_a": [1, 2]}, index=[1, 2]))

    def test__inner_join_multi_index(self):
        """given"""
        dfa = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
        dfa.columns = pd.MultiIndex.from_tuples([("a", "a")])

        dfb = pd.DataFrame({"a": [1, 2, 3]}, index=[1, 2, 3])

        """when"""
        df = dfa.inner_join(dfb, prefix='b')

        """then"""
        res = pd.DataFrame({"a": [2, 3], "b_a": [1, 2]}, index=[1, 2])
        res.columns = pd.MultiIndex.from_tuples([("a", "a"), ("b", "a")])
        pd.testing.assert_frame_equal(df, res)

    def test__cloc2(self):
        """given"""
        df = pd.DataFrame({"a": [2, 3], "b_a": [1, 2]}, index=[1, 2])
        df.columns = pd.MultiIndex.from_tuples([("a", "a"), ("b", "a")])

        """when"""
        single = df.cloc2("a")
        multi = df.cloc2(["a"])

        """then"""
        self.assertListEqual(single.columns.tolist(), ["a", "b"])
        self.assertListEqual(multi.columns.tolist(), [("a", "a"), ("a", "b")])
        self.assertIsInstance(multi.columns, pd.MultiIndex)


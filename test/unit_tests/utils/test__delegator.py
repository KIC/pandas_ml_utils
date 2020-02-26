from unittest import TestCase

import pandas as pd

from pandas_ml_utils.utils.delegator import delegate_to


class TestDelegator(TestCase):

    def test_delegator(self):
        """given"""
        class A(object):
            def __init__(self):
                self.a = "Aa"
                self.b = "Ab"

            @property
            def pa(self):
                return "Apa"

            @property
            def pb(self):
                return "Apb"

            def funcA(self, *args, **kwargs):
                return f'{args}, {kwargs}'

            def funcB(self, *args, **kwargs):
                return f'{args}, {kwargs}'

            @property
            def prop_new_A(self):
                return A()

            def func_new_A(self, *args, **kwargs):
                return A()

        @delegate_to("delegate", A)
        class SubA():
            def __init__(self, delegate):
                self.delegate = delegate # PrePostCaller(delegate)
                self.b = "Bb"

            @property
            def pb(self):
                return "Subpb"

            def funcB(self, *args, **kwargs):
                return "funcB"

        """when"""
        a = A()
        sub_a = SubA(a)

        """then"""
        # instance
        self.assertIsInstance(sub_a, SubA)
        # self.assertIsInstance(sub_a, A)

        # properties
        self.assertEqual(sub_a.pa, a.pa)
        self.assertEqual(sub_a.pa, "Apa")
        self.assertEqual(sub_a.pb, "Subpb")

        # methods
        self.assertEqual(sub_a.funcA("a", b="b"), a.funcA("a", b="b"))
        self.assertEqual(sub_a.funcA("a", b="b"), "('a',), {'b': 'b'}")
        self.assertEqual(sub_a.funcB("a", b="b"), "funcB")

        # wrapping function
        self.assertIsInstance(sub_a.prop_new_A, SubA)
        self.assertIsInstance(sub_a.func_new_A(1, a=2), SubA)

        # fields
        self.assertEqual(sub_a.b, "Bb")

    def test_delegator_with_pandas(self):
        """given"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}, index=["1", "2", "3"])

        @delegate_to("delegate", pd.DataFrame)
        class Sub(object):
            def __init__(self, delegate):
                self.delegate = delegate

            @property
            def values(self):
                return [1, 1, 1]


        """when"""
        sub_df = Sub(df)

        """then"""
        self.assertListEqual(sub_df.values, [1, 1, 1])
        self.assertIs(sub_df.columns, df.columns)
        self.assertIsInstance(sub_df.copy(), Sub)
        self.assertIsInstance(sub_df.loc[["1"]], Sub)

    def test_patched_data_frame(self):
        import pandas_ml_utils.utils.monkey_patched_dataframe as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}, index=["1", "2", "3"])

        print(df.values)



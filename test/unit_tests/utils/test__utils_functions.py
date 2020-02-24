from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ml_utils.utils.functions import call_callable_dynamic_args, integrate_nested_arrays


class TestUtilFunctions(TestCase):

    def test_call_dynamic_args(self):
        """when"""
        arguments = list(range(4))
        l1 = lambda a: a
        l2 = lambda a, b: a + b
        l3 = lambda a, *b: a + sum(b)
        l4 = lambda *a: sum(a)
        def f1(a, b, c, d):
            return a + b + c + d
        def f2(a, *b, **kwargs):
            return a + sum(b)
        def f3(a, b):
            return a + b

        """then"""
        self.assertEqual(call_callable_dynamic_args(l1, *arguments), 0)
        self.assertEqual(call_callable_dynamic_args(l2, *arguments), 1)
        self.assertEqual(call_callable_dynamic_args(l3, *arguments), 6)
        self.assertEqual(call_callable_dynamic_args(l4, *arguments), 6)
        self.assertEqual(call_callable_dynamic_args(f1, *arguments), 6)
        self.assertEqual(call_callable_dynamic_args(f2, *arguments), 6)
        self.assertEqual(call_callable_dynamic_args(f3, *arguments), 1)

    def test_call_dynamic_args_kwargs(self):
        """expect"""
        self.assertTrue(call_callable_dynamic_args(lambda a, b: True, 1, b=2))
        self.assertTrue(call_callable_dynamic_args(lambda a, b: True, a=1, b=2))
        self.assertTrue(call_callable_dynamic_args(lambda a, b: True, 1, 2))
        self.assertRaises(Exception, lambda: call_callable_dynamic_args(lambda a, b: True, 1))
        self.assertRaises(Exception, lambda: call_callable_dynamic_args(lambda a, b: True, 1, c=1))

    def test_inegrate_nested_array(self):
        """given"""
        x = np.array([1, 2])
        df = pd.DataFrame({"a": [np.zeros((4, 3)) for _ in range(10)],
                           "b": [np.ones((4, 3)) for _ in range(10)]})

        """when"""
        res1 = integrate_nested_arrays(df[["a"]].values)
        res2 = integrate_nested_arrays(df.values)
        res3 = integrate_nested_arrays(x)

        """then"""
        self.assertTrue(df.values[-1].dtype == 'object')
        self.assertEqual(res1.shape, (10, 4, 3))
        self.assertEqual(res2.shape, (10, 2, 4, 3))
        self.assertTrue(x is res3)
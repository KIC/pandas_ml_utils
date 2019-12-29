from unittest import TestCase

from pandas_ml_utils.utils.functions import call_callable_dyamic_args


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
        self.assertEqual(call_callable_dyamic_args(l1, *arguments), 0)
        self.assertEqual(call_callable_dyamic_args(l2, *arguments), 1)
        self.assertEqual(call_callable_dyamic_args(l3, *arguments), 6)
        self.assertEqual(call_callable_dyamic_args(l4, *arguments), 6)
        self.assertEqual(call_callable_dyamic_args(f1, *arguments), 6)
        self.assertEqual(call_callable_dyamic_args(f2, *arguments), 6)
        self.assertEqual(call_callable_dyamic_args(f3, *arguments), 1)

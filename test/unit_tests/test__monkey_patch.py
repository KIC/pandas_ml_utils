from unittest import TestCase

import pandas_ml_utils.monkey_patched_dataframe as pd
import numpy as np


class TestMonkeyPatch(TestCase):

    def test_patched_data_frame(self):
        """given"""
        df = pd.DataFrame(
            {
                ("A", "a"): [1, 2, 3, 4, 5],
                ("A", "b"): [3, 2, 1, 0, 0],
                ("A", "c"): [3, 2, 1, 0, 0],
                ("B", "a"): [1, 2, 3, 1, 2],
                ("B", "b"): [3, 2, 1, 0, 1],
                ("B", "c"): [3, 2, 1, 0, 1],
                ("C", "a"): [np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4))],
                ("C", "b"): [np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4))],
                ("C", "c"): [np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4))],
                ("D", "a"): [np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4))],
            },
            index=[1, 2, 3, 4, 5],
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns.tolist())

        """when"""
        print(df)
        rnnShape = df[["A"]].feature_values
        rnnShape2 = df[["A", "B"]].feature_values
        rnnShapeExt = df["C"].feature_values
        labelShape = df["D"].label_values

        """then"""
        print(rnnShape.shape, rnnShape2.shape, rnnShapeExt.shape, labelShape.shape)
        self.assertEqual((5, 3, 1), rnnShape.shape)
        self.assertEqual((5, 3, 2), rnnShape2.shape)
        self.assertEqual((5, 3, 2, 4), rnnShapeExt.shape)
        self.assertEqual((5, 2, 4), labelShape.shape)



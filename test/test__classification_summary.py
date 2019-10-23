import numpy as np
import pandas as pd
import logging

from unittest import TestCase
from sklearn.metrics import confusion_matrix
from pandas_ml_utils.classification.summary import ClassificationSummary


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestClassificationSummary(TestCase):

    def test_confusion_matrix_indices(self):
        index = np.array([1, 2, 3, 4, 5])
        pred = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        truth = np.array([True, True, True, False, False])

        cs = ClassificationSummary(truth, pred, index, None)
        cm = cs._confusion_matrix_indices()

        expected = np.array([[np.array([1, 3]), np.array([4])],
                             [np.array([2]), np.array([5])]])

        np.testing.assert_array_equal(cm[0,0], expected[0,0])
        np.testing.assert_array_equal(cm[0,1], expected[0,1])
        np.testing.assert_array_equal(cm[1,0], expected[1,0])
        np.testing.assert_array_equal(cm[1,1], expected[1,1])

        cm_skit = confusion_matrix(truth, pred)
        self.assertEqual(len(cm[0,0]), cm_skit[1,1])
        self.assertEqual(len(cm[0,1]), cm_skit[0,1])
        self.assertEqual(len(cm[1,0]), cm_skit[1,0])
        self.assertEqual(len(cm[1,1]), cm_skit[0,0])

        print(cs)

    def test_plot_backtest_no_fit(self):
        index = np.array([0, 1, 2, 3, 4])
        pred = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        truth = np.array([False, True, False, False, True])
        loss = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        cs = ClassificationSummary(truth, pred, index, pd.Series(loss))
        plot = cs.plot_backtest()
        self.assertIsNotNone(plot)

    def test_plot_backtest_a_fit(self):
        index = np.array([0, 1, 2, 3, 4])
        pred = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        truth = np.array([True, True, True, False, False])
        loss = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        cs = ClassificationSummary(truth, pred, index, pd.Series(loss))
        plot = cs.plot_backtest()
        self.assertIsNotNone(plot)

    def test_plot_backtest_perfect_fit(self):
        index = np.array([0, 1, 2, 3, 4])
        pred = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        truth = np.array([True, False, True, False, False])
        loss = np.array([1.0, 1.0, 1.0, 1.0, 1.0]) * -1

        cs = ClassificationSummary(truth, pred, index, pd.Series(loss))
        plot = cs.plot_backtest()
        self.assertIsNotNone(plot)
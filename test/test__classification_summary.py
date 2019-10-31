import numpy as np
import pandas as pd
import logging

from unittest import TestCase
from sklearn.metrics import confusion_matrix
from pandas_ml_utils.classification.summary import ClassificationSummary
from pandas_ml_utils.classification.classifier import _convert_probabilities

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

target = [0.0, 0.0, 0.0, 0.0, 0.0]
loss = np.array([1.0, 2.0, 4.0, 6.0, 8.0]) * -1

df = _convert_probabilities(pd.DataFrame({
    ("no fit", "target", "value"):          target,
    ("no fit", "prediction", "value"):      [1.0, 0.0, 1.0, 1.0, 0.0],
    ("no fit", "label", "value"):           [False, True, False, False, True],
    ("no fit", "loss", "value"):            loss,
    ("regular fit", "target", "value"):     target,
    ("regular fit", "prediction", "value"): [1.0, 0.0, 1.0, 1.0, 0.0],
    ("regular fit", "label", "value"):      [True, True, True, False, False],
    ("regular fit", "loss", "value"):       loss,
    ("perfect fit", "target", "value"):     target,
    ("perfect fit", "prediction", "value"): [1.0, 0.0, 1.0, 1.0, 0.0],
    ("perfect fit", "label", "value"):      [True, False, True, False, False],
    ("perfect fit", "loss", "value"):       loss,
}))


class TestClassificationSummary(TestCase):

    def test_confusion_matrix_indices(self):
        """given"""
        cs = ClassificationSummary(df)

        """when"""
        cm = cs.get_confusion_matrix()
        cl = cs.get_confusion_loss()

        """then"""
        np.testing.assert_array_equal(cm["no fit"], np.array([[0, 3], [2 ,0]]))
        np.testing.assert_array_equal(cm["regular fit"], np.array([[2, 1], [1, 1]]))
        np.testing.assert_array_equal(cm["perfect fit"], np.array([[2, 1], [0, 2]]))

        np.testing.assert_array_equal(cl["no fit"], np.array([[0., -11.], [-10, 0]]))
        np.testing.assert_array_equal(cl["regular fit"], np.array([[-5, -6.], [-2., -8.]]))
        np.testing.assert_array_equal(cl["perfect fit"], np.array([[-5., -6.], [0., -10]]))


    def test_plot_backtest_no_fit(self):
        index = np.array([0, 1, 2, 3, 4])
        pred = {"foo_target": np.array([1.0, 0.0, 1.0, 1.0, 0.0])}
        truth = np.array([False, True, False, False, True])
        loss = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        cs = ClassificationSummary(truth, pred, index, pd.Series(loss))
        plot = cs.plot_backtest()
        self.assertIsNotNone(plot)

    def test_plot_backtest_a_fit(self):
        index = np.array([0, 1, 2, 3, 4])
        pred = {"foo_target": np.array([1.0, 0.0, 1.0, 1.0, 0.0])}
        truth = np.array([True, True, True, False, False])
        loss = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        cs = ClassificationSummary(truth, pred, index, pd.Series(loss))
        plot = cs.plot_backtest()
        self.assertIsNotNone(plot)

    def test_plot_backtest_perfect_fit(self):
        index = np.array([0, 1, 2, 3, 4])
        pred = {"foo_target": np.array([1.0, 0.0, 1.0, 1.0, 0.0])}
        truth = np.array([True, False, True, False, False])
        loss = np.array([1.0, 1.0, 1.0, 1.0, 1.0]) * -1

        cs = ClassificationSummary(truth, pred, index, pd.Series(loss))
        plot = cs.plot_backtest()
        self.assertIsNotNone(plot)
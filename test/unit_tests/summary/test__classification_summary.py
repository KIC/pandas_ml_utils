import logging
from unittest import TestCase

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pandas_ml_utils.summary.binary_classification_summary import BinaryClassificationSummary
from pandas_ml_utils.constants import *
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

target = [0.0, 0.0, 0.0, 0.0, 0.0]
gross_loss = np.array([1.0, 2.0, 4.0, 6.0, 8.0]) * -1

df = pd.DataFrame({
    ("no fit", TARGET_COLUMN_NAME, "value"):          target,
    ("no fit", PREDICTION_COLUMN_NAME, "value"):      [1.0, 0.0, 1.0, 1.0, 0.0],
    ("no fit", LABEL_COLUMN_NAME, "value"):           [False, True, False, False, True],
    ("no fit", GROSS_LOSS_COLUMN_NAME, "value"):      gross_loss,
    ("regular fit", TARGET_COLUMN_NAME, "value"):     target,
    ("regular fit", PREDICTION_COLUMN_NAME, "value"): [1.0, 0.0, 1.0, 1.0, 0.0],
    ("regular fit", LABEL_COLUMN_NAME, "value"):      [True, True, True, False, False],
    ("regular fit", GROSS_LOSS_COLUMN_NAME, "value"): gross_loss,
    ("perfect fit", TARGET_COLUMN_NAME, "value"):     target,
    ("perfect fit", PREDICTION_COLUMN_NAME, "value"): [1.0, 0.0, 1.0, 1.0, 0.0],
    ("perfect fit", LABEL_COLUMN_NAME, "value"):      [True, False, True, False, False],
    ("perfect fit", GROSS_LOSS_COLUMN_NAME, "value"): gross_loss,
})


class TestClassificationSummary(TestCase):

    def test_confusion_matrix_indices(self):
        """given"""
        cs = BinaryClassificationSummary(df)

        """when"""
        cm = cs.get_confusion_matrix(total=False)
        cl = cs.get_confusion_loss(total=False)
        ms = cs.get_metrics()

        """then"""
        np.testing.assert_array_equal(cm[0], np.array([[0, 3], [2 ,0]]))
        np.testing.assert_array_equal(cm[1], np.array([[2, 1], [1, 1]]))
        np.testing.assert_array_equal(cm[2], np.array([[2, 1], [0, 2]]))

        np.testing.assert_array_equal(cl[0], np.array([[0., -11.], [-10, 0]]))
        np.testing.assert_array_equal(cl[1], np.array([[-5, -6.], [-2., -8.]]))
        np.testing.assert_array_equal(cl[2], np.array([[-5., -6.], [0., -10]]))

        np.testing.assert_array_almost_equal(np.array(list(ms.values())), np.array([1.25, 0.75, 0.48]), 2)

    def test_plots(self):
        """given"""
        cs = BinaryClassificationSummary(df)

        """when"""
        plots = cs.plot_classification()

        """then"""
        self.assertListEqual(list(plots.keys()), ['no fit', 'regular fit', 'perfect fit'])
        self.assertNotIn(None, plots.values())
        self.assertListEqual([type(p) for p in plots.values()], [Figure, Figure, Figure])

    def test_html(self):
        """given"""
        cs = BinaryClassificationSummary(df)

        """when"""
        html = cs._repr_html_()

        """then"""
        self.assertEqual(len(html), 154945)
from unittest import TestCase

import numpy as np

from pandas_ml_utils.sampling.boosting import KFoldBoostRareEvents, KEquallyWeightEvents
from pandas_ml_utils.utils.functions import unfold_parameter_space, one_hot


class TestBoosting(TestCase):

    def test_rare_events_boosting(self):
        """given"""
        x = np.arange(20)
        y = np.zeros(20)
        y[9] = True
        y[12] = True
        y[13] = True

        """when"""
        folds = KFoldBoostRareEvents(3, shuffle=True).split(x, y)

        """then"""
        for f, (train_idx, test_idx) in enumerate(folds):
            self.assertEqual(np.isin([9, 12, 13], train_idx).sum(), 3)

    def test_equal_weight_boosting(self):
        """given"""
        x = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9])
        y = np.array([one_hot(_x, 10) for _x in x])

        """when"""
        x_folds = KEquallyWeightEvents(3, 42).split(x, x)
        y_folds = KEquallyWeightEvents(3, 42).split(y, y)

        """then"""
        for f, (train_idx, test_idx) in enumerate(x_folds):
            print(f'\n{f}:\n{x[train_idx]}\n{x[test_idx]}')
            self.assertTrue((np.unique(x[train_idx], return_counts=True)[1] == 6).all())

        for f, (train_idx, test_idx) in enumerate(y_folds):
            self.assertTrue((np.unique(x[train_idx], return_counts=True)[1] == 6).all())

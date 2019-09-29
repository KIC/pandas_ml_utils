from unittest import TestCase

import numpy as np

from pandas_ml_utils.utils import unfold_parameter_space, KFoldBoostRareEvents


class TestUtils(TestCase):

    def test_unfold_parameter_space(self):
        """given"""
        parameter_space = {"a": range(3, 6), "b": [1, 2, 3], "c": [4, 5]}

        """when"""
        unfolded_parameter_space = unfold_parameter_space(parameter_space, {})

        """then"""
        self.assertEqual(len(unfolded_parameter_space), 3 * 3 * 2)

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

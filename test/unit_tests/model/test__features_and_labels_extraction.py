from unittest import TestCase

import pandas as pd
import numpy as np

from pandas_ml_utils.model.features_and_labels.features_and_labels_extractor import FeatureTargetLabelExtractor
from pandas_ml_utils.model.features_and_labels.features_and_labels import FeaturesAndLabels

DF = pd.DataFrame({"a": [1,2,3,4,5],
                   "b": [5,4,3,2,1],
                   "c": [1,2,3,4,5],
                   "d": [5,4,3,2,1],
                   "e": [5,4,3,2,1],})


class TestFeaturesAndLabelsExtraction(TestCase):

    def test_simple(self):
        """given"""
        fl = FeaturesAndLabels(["a"], ["d", "e"],
                               feature_lags=range(3),
                               targets={"b": None})

        """when"""
        df, f, l = FeatureTargetLabelExtractor(DF, fl).features_labels

        """then"""
        print(df.columns)
        print(f)
        np.testing.assert_array_almost_equal(f, np.array([[[3], [2], [1]],
                                                          [[4], [3], [2]],
                                                          [[5], [4], [3]]]))

    def test_pre_processor(self):
        """given"""
        fl = FeaturesAndLabels(["a"], ["b"], pre_processor=lambda _df, names: _df.rename(columns=names), a="lala")
        fl2 = fl.with_kwargs(a="lolo")

        """when"""
        df, _, _ = FeatureTargetLabelExtractor(DF, fl).features_labels
        df2, _, _ = FeatureTargetLabelExtractor(DF, fl2).features_labels

        """then"""
        self.assertListEqual(df.columns.tolist(), ['lala', 'b'])
        self.assertListEqual(df2.columns.tolist(), ['lolo', 'b'])

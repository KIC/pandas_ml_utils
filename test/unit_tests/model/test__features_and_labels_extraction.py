from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ml_utils.model.features_and_labels.features_and_labels import FeaturesAndLabels
from pandas_ml_utils.model.features_and_labels.features_and_labels_extractor import FeatureTargetLabelExtractor

DF = pd.DataFrame({"a": [1,2,3,4,5],
                   "b": [5,4,3,2,1],
                   "c": [1,2,3,4,5],
                   "d": [5,4,3,2,1],
                   "e": [5,4,3,2,1],})


class TestFeaturesAndLabelsExtraction(TestCase):

    def test_magic_arguments(self):
        """given"""
        labels = []

        def test(df, _labels):
            labels.append(_labels)
            return df

        """when"""
        FeatureTargetLabelExtractor(pd.DataFrame({}), FeaturesAndLabels(["a"], ["b"], pre_processor=test))

        """then"""
        self.assertListEqual(labels, [["b"]])

    def test_simple(self):
        """given"""
        fl = FeaturesAndLabels(["a"], ["d", "e"],
                               feature_lags=range(3),
                               targets={"b": None})

        """when"""
        f, l, _ = FeatureTargetLabelExtractor(DF, fl).features_labels_weights_df

        """then"""
        np.testing.assert_array_almost_equal(f.values, np.array([[[3], [2], [1]],
                                                                 [[4], [3], [2]],
                                                                 [[5], [4], [3]]]))

    def test_pre_processor(self):
        """given"""
        fl = FeaturesAndLabels(["lala"], ["b"], pre_processor=lambda df, **kwargs: df.rename(columns=kwargs), a="lala")

        """when"""
        f, l, w = FeatureTargetLabelExtractor(DF, fl).features_labels_weights_df

        """then"""
        self.assertEqual(len(f), len(l))
        self.assertListEqual(f.columns.tolist(), ['lala'])
        self.assertListEqual(l.columns.tolist(), ['b'])

    def test_wither(self):
        """given"""
        fl = FeaturesAndLabels(["lala"], ["b"],
                               feature_lags=[0, 1],
                               pre_processor=lambda df, names: df.rename(columns=names), a="lala")

        """when"""
        fl = fl.with_kwargs(a="lolo", b="lala")

        """then"""
        self.assertListEqual(fl.feature_lags, [0, 1])
        self.assertEqual(fl.kwargs["a"], "lolo")
        self.assertEqual(fl.kwargs["b"], "lala")


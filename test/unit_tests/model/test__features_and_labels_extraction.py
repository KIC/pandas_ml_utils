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
        df, f, l = FeatureTargetLabelExtractor(DF, fl).features_labels

        """then"""
        print(df.columns)
        print(f)
        np.testing.assert_array_almost_equal(f, np.array([[[3], [2], [1]],
                                                          [[4], [3], [2]],
                                                          [[5], [4], [3]]]))

    def test_pre_processor(self):
        """given"""
        fl = FeaturesAndLabels(["lala"], ["b"], pre_processor=lambda _df, **kwargs: _df.rename(columns=kwargs), a="lala")

        """when"""
        df, _, _ = FeatureTargetLabelExtractor(DF, fl).features_labels

        """then"""
        self.assertListEqual(df.columns.tolist(), ['lala', 'b'])

    def test_wither(self):
        """given"""
        fl = FeaturesAndLabels(["lala"], ["b"],
                               feature_lags=[0, 1],
                               pre_processor=lambda _df, names: _df.rename(columns=names), a="lala")

        """when"""
        fl = fl.with_kwargs(a="lolo", b="lala")

        """then"""
        self.assertListEqual(fl.feature_lags, [0, 1])
        self.assertEqual(fl.kwargs["a"], "lolo")
        self.assertEqual(fl.kwargs["b"], "lala")


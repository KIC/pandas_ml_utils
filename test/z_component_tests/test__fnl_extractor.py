import logging
import os
import unittest

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

import pandas_ml_utils as pdu
from pandas_ml_utils.constants import *
from pandas_ml_utils.summary.binary_classification_summary import BinaryClassificationSummary
from pandas_ml_utils.utils.functions import fig_to_png_base64

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test.csv")


class FeatureAndLabelsExtractorTest(unittest.TestCase):

    def test_extractor(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')

        """when"""
        extractor = df.features_and_label_extractor(
            pdu.SkitModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), alpha=0.001, random_state=42),
                pdu.FeaturesAndLabels(features=['feature'], labels=['label'],
                                      loss=lambda df: df["spy_Close"] - df["spy_Open"],
                                      targets=lambda df: df["spy_Close"],
                                      pre_processor=lambda _df, _: pdu.LazyDataFrame(
                                          _df,
                                          feature=lambda f: f["vix_Close"].rolling(2).mean(),
                                          label=lambda f: (f["spy_Close"].shift(1) > f["spy_Open"]).shift(-1)))))

        fnl = extractor.features_labels

        """then"""
        self.assertListEqual(extractor.features_df.columns.tolist(), ['feature'])
        self.assertListEqual(extractor.labels_df.columns.tolist(), ['label'])
        self.assertListEqual(extractor.source_df[SOURCE_COLUMN_NAME].columns.tolist(), [*df.columns.tolist(), 'feature', 'label'])
        self.assertListEqual(extractor.target_df.columns.tolist(), [(TARGET_COLUMN_NAME, "spy_Close")])
        self.assertListEqual(extractor.loss_df.columns.tolist(), [(LOSS_COLUMN_NAME, "loss")])
        self.assertEqual(fnl[0].shape, (6704, 2))
        self.assertEqual(fnl[1].shape, (6704, 1))
        self.assertEqual(fnl[2].shape, (6704, 1))


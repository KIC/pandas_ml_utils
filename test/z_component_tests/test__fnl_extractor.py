import logging
import unittest

from sklearn.neural_network import MLPClassifier
from test.config import TEST_FILE

from pandas_ml_utils import pd, SkModel, FeaturesAndLabels, LazyDataFrame
from pandas_ml_utils.constants import *

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class FeatureAndLabelsExtractorTest(unittest.TestCase):

    def test_extractor(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')

        """when"""
        extractor = df.features_and_label_extractor(
            SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), alpha=0.001, random_state=42),
                FeaturesAndLabels(features=['feature'], labels=['label'],
                                  gross_loss=lambda df: df["spy_Close"] - df["spy_Open"],
                                  targets=lambda df: df["spy_Close"],
                                  pre_processor=lambda _df: LazyDataFrame(
                                          _df,
                                          feature=lambda f: f["vix_Close"].rolling(2).mean(),
                                          label=lambda f: (f["spy_Close"].shift(1) > f["spy_Open"]).shift(-1)))))

        fnl = extractor.features_labels_weights_df

        """then"""
        self.assertListEqual(extractor.features_df.columns.tolist(), ['feature'])
        self.assertListEqual(extractor.labels_df.columns.tolist(), ['label'])
        self.assertListEqual(extractor.source_df[SOURCE_COLUMN_NAME].columns.tolist(), [*df.columns.tolist(), 'feature', 'label'])
        self.assertListEqual(extractor.target_df.columns.tolist(), [(TARGET_COLUMN_NAME, "spy_Close")])
        self.assertListEqual(extractor.gross_loss_df.columns.tolist(), [(GROSS_LOSS_COLUMN_NAME, GROSS_LOSS_COLUMN_NAME)])
        self.assertEqual(fnl[0].shape, (6704, 1))
        self.assertEqual(fnl[1].shape, (6704, 1))
        self.assertIsNone(fnl[2])


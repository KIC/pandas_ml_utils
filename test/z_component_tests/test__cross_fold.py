import logging
import unittest

from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from test.config import TEST_FILE

from pandas_ml_utils import pd, SkModel, FeaturesAndLabels

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class CrossFoldTest(unittest.TestCase):

    def test_binary_classification_kfold(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df['label'] = df["spy_Close"] > df["spy_Open"]

        """and cross validation"""
        called = False
        cv = KFold(2)

        def split(x, y=None, group=None):
            nonlocal called
            called = True
            return cv.split(x, y, group)

        """when"""
        fit = df.fit(
            SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), alpha=0.001, random_state=42),
                FeaturesAndLabels(features=['vix_Close'], labels=['label'],
                                  gross_loss=lambda df: df["spy_Close"] - df["spy_Open"])),
            test_size=0.4,
            cross_validation=(1, split),
            test_validate_split_seed=42)

        """then"""
        # no exception thrown
        self.assertTrue(called)

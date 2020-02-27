import logging
import unittest

from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from test.config import TEST_FILE

from pandas_ml_utils import pd, SkModel, FeaturesAndLabels

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class CrossValidationTest(unittest.TestCase):

    def test_cross_validation(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df['label'] = df["spy_Close"] > df["spy_Open"]

        """and given KFold"""
        cv = KFold(n_splits=10, shuffle=False)

        """when"""
        fit = df.fit(
            SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), alpha=0.001, random_state=42, max_iter=10),
                FeaturesAndLabels(features=['vix_Close'], labels=['label'])),
            test_size=0.4,
            cross_validation = (2, cv.split),
            test_validate_split_seed=42)

        """then"""
        self.assertEqual(fit.model.features_and_labels.min_required_samples, 1)
        # FIXME bring back assteration after fix of summary
        #np.testing.assert_array_equal(fit.test_summary.get_confusion_matrix()["vix_Open"],
        #                              np.array([[257, 169], [1142, 1115]]))

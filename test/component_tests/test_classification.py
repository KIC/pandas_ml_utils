import logging
import os
import unittest

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

import pandas_ml_utils as pdu

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test.csv")


class ClassificationTest(unittest.TestCase):

    def test_binary_classification(self):
        import talib

        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["sma"] = talib.SMA(df["spy_Close"])
        df["is_above"] = (df["spy_Close"] / df["sma"]) > 1

        model = pdu.SkitModel(
            MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
            pdu.FeaturesAndLabels(features=['vix_Close'],
                                  labels=["is_above"],
                                  targets=lambda _, frame: frame["sma"],
                                  loss=lambda _, frame: frame["spy_Close"] - frame["sma"]))

        """when"""
        fit = df.fit(model, test_size=0.4, test_validate_split_seed=42)
        result = fit.training_summary.target_data

        """then"""
        self.assertListEqual(result.columns.tolist(), [('prediction', 'is_above'), ('label', 'is_above'), ('loss', 'loss'), ('target', 'sma')])
        self.assertEqual(len(result), 4023)

    def test_multiple_classifications(self):
        from pandas_ml_utils.model.features_and_labels_utils.target_encoder import OneHotEncodedTargets
        import talib

        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["sma"] = talib.SMA(df["spy_Close"])
        df["label"] = df["spy_Close"] / df["sma"] -1

        def make_targets(t, frame):
            space = np.array([-1, -0.05, 0.5, 1])
            res = frame.apply(lambda x: x["sma"] - space, axis=1,
                              result_type='expand')
            res.columns = ["close <0.1", "close <0.05", "close >0", "close >0.05"]
            return res

        model = pdu.SkitModel(
            MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
            pdu.FeaturesAndLabels(features=['vix_Close'],
                                  labels=OneHotEncodedTargets("label", np.linspace(-0.1, 0.1, 5, endpoint=True)),
                                  targets=make_targets))

        """when"""
        fit = df.fit(model, test_size=0.4, test_validate_split_seed=42,)
        result = fit.training_summary.target_data

        """then"""
        print(result.columns.tolist())
        self.assertListEqual(result.columns.tolist(),
                             [('prediction', 'label #0'), ('prediction', 'label #1'), ('prediction', 'label #2'), ('prediction', 'label #3'),
                              ('label', 'label #0'), ('label', 'label #1'), ('label', 'label #2'), ('label', 'label #3'),
                              ('target', 'close <0.1'), ('target', 'close <0.05'), ('target', 'close >0'), ('target', 'close >0.05')]
)
        #self.assertEqual(len(result["sma"]), 4023)

    def test_multi_model_multiple_classifications(self):
        # FIXME implement this
        from pandas_ml_utils.model.features_and_labels_utils.target_encoder import OneHotEncodedTargets
        import talib

        df = pd.read_csv(f'{__name__}.csv', index_col='Date')
        df["sma"] = talib.SMA(df["spy_Close"])
        df["label_1"] = df["spy_Close"] / df["sma"] - 1
        df["label_2"] = df["spy_Close"] / df["sma"] - 2

        model = pdu.MultiModel(
            pdu.SkitModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
                          pdu.FeaturesAndLabels(features=['vix_Close'],
                                                labels={f"-{t}": OneHotEncodedTargets("label", np.linspace(-0.1, 0.1, 3, endpoint=True)) for t in range(1, 3)})))

        fit = df.fit_classifier(model, test_size=0.4, test_validate_split_seed=42,)

        self.assertTrue(False)



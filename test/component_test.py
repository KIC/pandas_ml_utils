import pandas as pd
import numpy as np
import unittest
import pandas_ml_utils as pdu

from sklearn.neural_network import MLPClassifier

import logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class ComponentTest(unittest.TestCase):

    def test_fetch_yahoo(self):
        df = pd.fetch_yahoo(spy="SPY").tail()
        self.assertTrue(df["spy_Close"].sum() > 0)

    def test_fit_classifier_full(self):
        df = pd.read_csv(f'{__name__}.csv', index_col='Date')
        df['label'] = df["spy_Close"] > df["spy_Open"]

        # fit
        fit = df.fit_classifier(pdu.SkitModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), alpha=0.001,
                                                            random_state=42),
                                              pdu.FeaturesAndLabels(features=['vix_Close'], labels=['label'], target_columns=["vix_Open"], loss_column="spy_Volume")),
                                test_size=0.4,
                                test_validate_split_seed=42)

        self.assertEqual(fit.model.min_required_data, 1)
        np.testing.assert_array_equal(fit.test_summary.confusion_count(), np.array([[744, 586],
                                                                                    [655, 698]]))

        # backtest
        fitted_model = fit.model
        backtest_classification = df.backtest(fitted_model)
        np.testing.assert_array_equal(backtest_classification.confusion_count(), np.array([[1811, 1458],
                                                                                           [1657, 1780]]))

        # classify
        fitted_model = fit.model
        classified_df = df.classify(fitted_model)
        print(classified_df.tail())

        self.assertEqual(len(classified_df[classified_df["prediction"] == False]), 3437)
        self.assertTrue(classified_df["loss"].sum() > 0)
        self.assertListEqual(classified_df.columns.tolist(),
                             ["vix_Close", "traget_vix_Open", "loss", "prediction", "prediction_proba"])

        # classify tail
        fitted_model = fit.model
        classified_df = df.classify(fitted_model, 2)
        self.assertEqual(len(classified_df), 2)

    def test_fit_classifier_simple(self):
        df = pd.read_csv(f'{__name__}.csv', index_col='Date')
        df['label'] = df["spy_Close"] > df["spy_Open"]

        # fit
        fit = df.fit_classifier(pdu.SkitModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), alpha=0.001,
                                                            random_state=42),
                                              pdu.FeaturesAndLabels(features=['vix_Close'], labels=['label'])),
                                test_size=0.4,
                                test_validate_split_seed=42)

        fitted_model = fit.model
        classified_df = df.classify(fitted_model)

        self.assertListEqual(classified_df.columns.tolist(),
                             ["vix_Close", "prediction", "prediction_proba"])

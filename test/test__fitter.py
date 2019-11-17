from unittest import TestCase

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC

from pandas_ml_utils.model.fitter import _fit, _backtest, _predict
from pandas_ml_utils.model.models import *

df = pd.DataFrame({"a": np.array([0.1, 0.01]), "b": np.array([True, False]), "c": np.array([False, True])})

class TestFitter(TestCase):

    def test__fit(self):
        """given"""
        features_and_labels = FeaturesAndLabels(["a"], ["b"])
        providers = [
            SkitModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(1, 1), alpha=0.001, random_state=42),
                      features_and_labels, foo='bar'),
            SkitModel(LogisticRegression(), features_and_labels),
            SkitModel(LinearSVC(), features_and_labels),
            SkitModel(RandomForestClassifier(), features_and_labels)]

        """when"""
        fitts = [_fit(df, p, 0)[1][0] for p in providers]
        fits_df_columns = [f.columns.tolist() for f in fitts]

        """then"""
        expected_columns = [('target', 'target', 'value'), ('target', 'prediction', 'value'), ('target', 'label', 'value'), ('target', 'loss', 'value')]
        self.assertListEqual(fits_df_columns[0], expected_columns)
        self.assertListEqual(fits_df_columns[1], expected_columns)
        self.assertListEqual(fits_df_columns[2], expected_columns)
        self.assertListEqual(fits_df_columns[3], expected_columns)
        np.testing.assert_array_equal(fitts[0]["target", "label", "value"].values, df["b"].values)
        np.testing.assert_array_equal(fitts[1]["target", "label", "value"].values, df["b"].values)
        np.testing.assert_array_equal(fitts[2]["target", "label", "value"].values, df["b"].values)
        np.testing.assert_array_equal(fitts[3]["target", "label", "value"].values, df["b"].values)

    def test__backtest(self):
        """given"""
        fls = [FeaturesAndLabels(["a"], ["b"]),
               FeaturesAndLabels(["a"], ["b"], targets="b"),
               FeaturesAndLabels(["a"], ["b", "c"], targets={"b": (-1, ["b", "c"])}),
               FeaturesAndLabels(["a"], ["b", "c"], targets={"b": (-1, ["b", "c"]), "a": (-2, ["b", "c"])})]

        providers = [SkitModel(MLPRegressor(activation='tanh', hidden_layer_sizes=(1, 1), alpha=0.001, random_state=42),
                               features_and_labels=fl) for fl in fls]

        """when"""
        fitted_models = [_fit(df, p, 0)[0] for p in providers]
        backtests = [_backtest(df, fm) for fm in fitted_models]
        backtest_columns = [b.columns.tolist() for b in backtests]

        """then"""
        # print(backtest_columns[3])
        self.assertEqual(backtest_columns[0], [('target', 'target', 'value'), ('target', 'prediction', 'value'), ('target', 'label', 'value'), ('target', 'loss', 'value')])
        self.assertEqual(backtest_columns[1], [('b', 'target', 'value'), ('b', 'prediction', 'value'), ('b', 'label', 'value'), ('b', 'loss', 'value')])
        self.assertEqual(backtest_columns[2], [('b', 'target', 'value'), ('b', 'prediction', 'b'), ('b', 'prediction', 'c'), ('b', 'label', 'b'), ('b', 'label', 'c'), ('b', 'loss', 'value')])
        self.assertEqual(backtest_columns[3], [('b', 'target', 'value'), ('b', 'prediction', 'b'), ('b', 'prediction', 'c'), ('a', 'target', 'value'), ('a', 'prediction', 'b'), ('a', 'prediction', 'c'), ('b', 'label', 'b'), ('b', 'label', 'c'), ('a', 'label', 'b'), ('a', 'label', 'c'), ('b', 'loss', 'value'), ('a', 'loss', 'value')])
        np.testing.assert_array_equal(backtests[3]["b", "label", "b"].values, df["b"].values)
        np.testing.assert_array_equal(backtests[3]["a", "label", "b"].values, df["b"].values)
        np.testing.assert_array_equal(backtests[3]["b", "label", "c"].values, df["c"].values)
        np.testing.assert_array_equal(backtests[3]["a", "label", "c"].values, df["c"].values)
        np.testing.assert_array_equal(backtests[3]["a", "loss", "value"].values, -2)
        np.testing.assert_array_equal(backtests[3]["b", "loss", "value"].values, -1)

    def test__predict(self):
        """given"""
        fls = [FeaturesAndLabels(["a"], ["b"]),
               FeaturesAndLabels(["a"], ["b"], targets="b"),
               FeaturesAndLabels(["a"], ["c", "b"], targets={"b": (-1, ["b", "c"])}),
               FeaturesAndLabels(["a"], ["c", "b"], targets={"b": (-1, ["b", "c"]), "a": (-1, ["b", "c"])})]

        providers = [SkitModel(MLPRegressor(activation='tanh', hidden_layer_sizes=(1, 1), alpha=0.001, random_state=42),
                               features_and_labels=fl) for fl in fls]


        """when"""
        fitted_models = [_fit(df, p, 0)[0] for p in providers]

        """then"""
        predictions = [_predict(df, fm) for fm in fitted_models]
        print(predictions[-1].columns.tolist())
        self.assertEqual(predictions[0].columns.tolist(), [('target', 'target', 'value'), ('target', 'prediction', 'value')])
        self.assertEqual(predictions[1].columns.tolist(), [('b', 'target', 'value'), ('b', 'prediction', 'value')])
        self.assertEqual(predictions[2].columns.tolist(), [('b', 'target', 'value'), ('b', 'prediction', 'b'), ('b', 'prediction', 'c')])
        self.assertEqual(predictions[3].columns.tolist(), [('b', 'target', 'value'), ('b', 'prediction', 'b'), ('b', 'prediction', 'c') , ('a', 'target', 'value'), ('a', 'prediction', 'b'), ('a', 'prediction', 'c')])

    def test__predict_with_lags(self):
        """given"""
        df = pd.DataFrame({"a": [0.5592344, 0.60739384, 0.19994533, 0.56642537, 0.50965677,
                                 0.168989, 0.94080671, 0.76651769, 0.8403563, 0.4003567,
                                 0.24295908, 0.50706317, 0.66612371, 0.4020924, 0.21776017,
                                 0.32559497, 0.12721287, 0.13904584, 0.65887554, 0.08830925],
                           "b": range(20)})

        fl = FeaturesAndLabels(["a"], ["b"], feature_lags=[0,1,2])
        provider = SkitModel(MLPRegressor(activation='tanh', hidden_layer_sizes=(1, 1), alpha=0.001, random_state=42),
                             features_and_labels=fl)

        """when"""
        model, summaries, trails = _fit(df, provider, 0)
        backtest = _backtest(df, model)

        """then"""
        predictions = _predict(df, model)
        self.assertListEqual(predictions.columns.tolist(), [('target', 'target', 'value'), ('target', 'prediction', 'value')])
        self.assertEqual(fl.min_required_samples, 3)

        self.assertListEqual(backtest.columns.tolist(), [('target', 'target', 'value'), ('target', 'prediction', 'value'), ('target', 'label', 'value'), ('target', 'loss', 'value')])
        self.assertEqual(backtest["target", "loss", "value"].sum(), -18)
        self.assertEqual(backtest["target", "label", "value"].sum(), 189)

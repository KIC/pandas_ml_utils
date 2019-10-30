import pandas as pd
from unittest import TestCase

from pandas_ml_utils.model.models import *
from pandas_ml_utils.model.fitter import _fit, _backtest, _predict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC

features_and_labels = FeaturesAndLabels(["a"], ["b"])

df = pd.DataFrame({"a": np.array([0.1, 0.01]), "b": np.array([True, False])})

class TestFitter(TestCase):

    def test__fit(self):
        """given"""
        providers = [
            SkitModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(1, 1), alpha=0.001, random_state=42),
                      features_and_labels, foo='bar'),
            SkitModel(LogisticRegression(), features_and_labels),
            SkitModel(LinearSVC(), features_and_labels),
            SkitModel(RandomForestClassifier(), features_and_labels)]

        """when"""
        fitts = [_fit(df, p, 0) for p in providers]

        """then"""
        print(fitts) # FIXME test something

    def test__backtest(self):
        """given"""
        fls = [FeaturesAndLabels(["a"], ["b"]),
               FeaturesAndLabels(["a"], ["b"], targets="b"),
               FeaturesAndLabels(["a"], ["a", "b"], targets={"b": (-1, ["b", "b"])}),
               FeaturesAndLabels(["a"], ["a", "b"], targets={"b": (-1, ["b", "b"]), "a": (-1, ["b", "b"])})]

        providers = [SkitModel(MLPRegressor(activation='tanh', hidden_layer_sizes=(1, 1), alpha=0.001, random_state=42),
                               features_and_labels=fl) for fl in fls]

        """when"""
        fitted_models = [_fit(df, p, 0)[0] for p in providers]
        backtest_columns = [_backtest(df, fm).columns.tolist() for fm in fitted_models]

        """then"""
        print(backtest_columns)
        self.assertEqual(backtest_columns[0], ['target', 'prediction', 'label', 'loss', 'feature_a'])
        self.assertEqual(backtest_columns[1], ['target_b', 'prediction_b', 'label_b', 'loss', 'feature_a'])
        self.assertEqual(backtest_columns[2], ['target_b', 'prediction_b_a', 'prediction_b_b', 'label_b_b', 'loss', 'feature_a'])
        self.assertEqual(backtest_columns[3], ['target_b', 'target_a', 'prediction_b_a', 'prediction_b_b', 'prediction_a_a', 'prediction_a_b', 'label_b_b', 'label_a_b', 'loss_b', 'loss_a', 'feature_a'])

    def test__predict(self):
        """given"""
        fls = [FeaturesAndLabels(["a"], ["b"]),
               FeaturesAndLabels(["a"], ["b"], targets="b"),
               FeaturesAndLabels(["a"], ["a", "b"], targets={"b": (-1, ["b", "b"])}),
               FeaturesAndLabels(["a"], ["a", "b"], targets={"b": (-1, ["b", "b"]), "a": (-1, ["b", "b"])})]

        providers = [SkitModel(MLPRegressor(activation='tanh', hidden_layer_sizes=(1, 1), alpha=0.001, random_state=42),
                               features_and_labels=fl) for fl in fls]


        """when"""
        fitted_models = [_fit(df, p, 0)[0] for p in providers]

        """then"""
        predictions = [_predict(df, fm) for fm in fitted_models]
        print(predictions[-1].columns.tolist())
        self.assertEqual(predictions[0].columns.tolist(), ["a", "target", "prediction"])
        self.assertEqual(predictions[1].columns.tolist(), ["a", "target_b", "prediction_b"])
        self.assertEqual(predictions[2].columns.tolist(), ["a", "target_b", "prediction_b_a", "prediction_b_b"])




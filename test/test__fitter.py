import unittest
import pandas as pd
from unittest import TestCase

from keras.optimizers import Adam, RMSprop

from pandas_ml_utils.model.features_and_Labels import FeaturesAndLabels
from pandas_ml_utils.model.models import *
from pandas_ml_utils.model.fitter import _fit, _backtest, _predict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import BaseLogger

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
        pass

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
        self.assertEqual(predictions[2].columns.tolist(), ["a", "target_b", "loss", "prediction_b_a", "prediction_b_b"])




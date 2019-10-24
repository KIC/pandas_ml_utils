from unittest import TestCase

from pandas_ml_utils.model.features_and_Labels import FeaturesAndLabels
from pandas_ml_utils.model.models import *
from sklearn.neural_network import MLPClassifier

features_and_labels = FeaturesAndLabels([], [])


class TestModel(TestCase):

    def test_skit_model(self):
        """given"""
        model_provider = SkitModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(60,50), alpha=0.001, random_state=42), features_and_labels)

        """when"""
        model1 = model_provider()
        model2 = model_provider(activation='logistic')

        """then"""
        self.assertEqual(model1.skit_model.activation, 'tanh')
        self.assertEqual(model2.skit_model.activation, 'logistic')

    def test_keras_model(self):
        # TODO make a test
        self.assertTrue(True)

from unittest import TestCase

from keras.optimizers import Adam, RMSprop

from pandas_ml_utils.model.features_and_Labels import FeaturesAndLabels
from pandas_ml_utils.model.models import *
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import BaseLogger

features_and_labels = FeaturesAndLabels([], [])


class TestModel(TestCase):

    def test_skit_model(self):
        """given"""
        model_provider = SkitModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(60,50), alpha=0.001, random_state=42), features_and_labels, foo='bar')

        """when"""
        model1 = model_provider()
        model2 = model_provider(activation='logistic')

        """then"""
        self.assertEqual(model1.kwargs, model2.kwargs)
        self.assertFalse(model1.kwargs is model2.kwargs)
        self.assertEqual(model1.skit_model.activation, 'tanh')
        self.assertEqual(model2.skit_model.activation, 'logistic')

    def test_keras_model(self):
        """given"""
        def keras_model_provider(optimizer='adam'):
            model = Sequential()
            model.add(Dense(10))
            model.compile(optimizer)
            return model

        """when"""
        model_provider = KerasModel(keras_model_provider, features_and_labels, callbacks=[BaseLogger()], foo='bar')
        model1 = model_provider()
        model2 = model_provider(optimizer='rmsprop')

        """then"""
        self.assertEqual(model1.kwargs, model2.kwargs)
        self.assertFalse(model1.kwargs is model2.kwargs)
        self.assertEqual(len(model1.callbacks), len(model2.callbacks))
        self.assertEqual(type(model1.callbacks[0]), type(model2.callbacks[0]))
        self.assertNotEqual(model1.callbacks[0], model2.callbacks[0])
        self.assertEqual(type(model1.keras_model.optimizer), Adam)
        self.assertEqual(type(model2.keras_model.optimizer), RMSprop)
        self.assertTrue(True)

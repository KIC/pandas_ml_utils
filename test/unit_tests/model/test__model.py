import operator
from unittest import TestCase

import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from pandas_ml_utils import LazyDataFrame
from pandas_ml_utils.model.models import *

features_and_labels = FeaturesAndLabels([], [])


class TestModel(TestCase):

    def test_reshape_rnn_as_ar(self):
        np.testing.assert_array_almost_equal(SkModel.reshape_rnn_as_ar(np.array([[[1], [2]]], ndmin=3)),
                                             np.array([[1, 2]], ndmin=2))

        np.testing.assert_array_almost_equal(SkModel.reshape_rnn_as_ar(np.array([[1, 2]], ndmin=2)),
                                             np.array([[1, 2]], ndmin=2))

    def test_different_skit_models(self):
        """given"""
        providers = [SkModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(1, 1), alpha=0.001, random_state=42), features_and_labels, foo='bar'),
                     SkModel(LogisticRegression(), features_and_labels),
                     SkModel(LinearSVC(), features_and_labels),
                     SkModel(RandomForestClassifier(), features_and_labels)
                     ]

        """when"""
        losses = [model.fit(np.array([[0.1], [0.01]]), np.array([True, False]), np.array([[0.1], [0.01]]),
                            np.array([True, False]),
                            [0, 1], [2, 3]) for model in providers]

        """then"""
        for loss in losses:
            self.assertIsNotNone(loss)

    def test_skit_model(self):
        """given"""
        model_provider = SkModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(1, 1), alpha=0.001, random_state=42), features_and_labels, foo='bar')

        """when"""
        model1 = model_provider()
        model2 = model_provider(activation='logistic')

        loss = model2.fit(np.array([[0.1], [0.01]]), np.array([True, False]), np.array([[0.1], [0.01]]), np.array([True, False]),
                          [0, 1], [2, 3])

        """then"""
        self.assertIsNotNone(loss)
        self.assertEqual(model1.kwargs, model2.kwargs)
        self.assertFalse(model1.kwargs is model2.kwargs)
        self.assertEqual(model1.skit_model.activation, 'tanh')
        self.assertEqual(model2.skit_model.activation, 'logistic')

    def test_keras_model(self):
        """prepare environment to be non cuda"""
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from keras.callbacks import BaseLogger
        from keras.layers import Dense
        from keras.models import Sequential
        from keras.optimizers import Adam, RMSprop

        """given"""
        def keras_model_provider(optimizer='adam'):
            model = Sequential()
            model.add(Dense(1, input_dim=1))
            model.compile(optimizer, loss='mse')
            return model

        """when"""
        model_provider = KerasModel(keras_model_provider, features_and_labels, callbacks=[BaseLogger], verbose=0, foo="bar")
        model1 = model_provider()
        model2 = model_provider(optimizer='rmsprop')

        """then weighs must be equal"""
        np.testing.assert_array_almost_equal(model1.get_weights(), model2.get_weights())

        """and after we fit one model"""
        loss = model2.fit(np.array([0.1, 0.01]), np.array([0.1, 0.01]), np.array([0.1, 0.01]), np.array([0.1, 0.01]),
                          [0,1], [2,3])

        """then"""
        self.assertIsNotNone(loss)
        self.assertEqual({**model1.kwargs, "optimizer": 'rmsprop'}, model2.kwargs)
        self.assertFalse(model1.kwargs is model2.kwargs)
        self.assertEqual(len(model1.callbacks), len(model2.callbacks))
        self.assertEqual(model1.callbacks[0], model2.callbacks[0])
        self.assertNotEqual(model1.session, model2.session)
        self.assertNotEqual(model1.graph, model2.graph)
        self.assertEqual(type(model1.keras_model.optimizer), Adam)
        self.assertEqual(type(model2.keras_model.optimizer), RMSprop)
        np.testing.assert_array_almost_equal(model2.get_weights(), model2().get_weights())
        np.testing.assert_array_compare(operator.__ne__, model1.get_weights(), model2.get_weights())

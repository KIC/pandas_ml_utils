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
        np.testing.assert_array_almost_equal(SkitModel.reshape_rnn_as_ar(np.array([[[1], [2]]], ndmin=3)),
                                             np.array([[1, 2]], ndmin=2))

        np.testing.assert_array_almost_equal(SkitModel.reshape_rnn_as_ar(np.array([[1, 2]], ndmin=2)),
                                             np.array([[1, 2]], ndmin=2))

    def test_different_skit_models(self):
        """given"""
        providers = [SkitModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(1,1), alpha=0.001, random_state=42), features_and_labels, foo='bar'),
                     SkitModel(LogisticRegression(), features_and_labels),
                     SkitModel(LinearSVC(), features_and_labels),
                     SkitModel(RandomForestClassifier(), features_and_labels)
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
        model_provider = SkitModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(1,1), alpha=0.001, random_state=42), features_and_labels, foo='bar')

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
        model_provider = KerasModel(keras_model_provider, features_and_labels, callbacks=[BaseLogger()], verbose=0, foo="bar")
        model1 = model_provider()
        model2 = model_provider(optimizer='rmsprop')

        loss = model2.fit(np.array([0.1, 0.01]), np.array([0.1, 0.01]), np.array([0.1, 0.01]), np.array([0.1, 0.01]),
                          [0,1], [2,3])

        """then"""
        self.assertIsNotNone(loss)
        self.assertEqual(model1.kwargs, model2.kwargs)
        self.assertFalse(model1.kwargs is model2.kwargs)
        self.assertEqual(len(model1.callbacks), len(model2.callbacks))
        self.assertEqual(type(model1.callbacks[0]), type(model2.callbacks[0]))
        self.assertNotEqual(model1.callbacks[0], model2.callbacks[0])
        self.assertEqual(type(model1.keras_model.optimizer), Adam)
        self.assertEqual(type(model2.keras_model.optimizer), RMSprop)
        self.assertTrue(True)

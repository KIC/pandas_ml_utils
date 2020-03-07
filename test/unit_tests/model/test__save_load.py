from unittest import TestCase

import numpy as np
from keras import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from pandas_ml_utils import pd, SkModel, FeaturesAndLabels, LazyDataFrame, MultiModel, KerasModel, Model

df = pd.DataFrame({"a": [0.5592344, 0.60739384, 0.19994533, 0.56642537, 0.50965677,
                         0.168989, 0.94080671, 0.76651769, 0.8403563, 0.4003567,
                         0.24295908, 0.50706317, 0.66612371, 0.4020924, 0.21776017,
                         0.32559497, 0.12721287, 0.13904584, 0.65887554, 0.08830925],
                   "b": np.random.randint(2, size=20)})


class TestSaveLoad(TestCase):

    def test_save_load_models(self):
        """given"""
        features_and_labels = FeaturesAndLabels(["a"], ["b"])

        def keras_model_provider(optimizer='adam'):
            model = Sequential()
            model.add(Dense(1, input_dim=1, activation='sigmoid'))
            model.compile(optimizer, loss='mse')
            return model

        providers = [
            SkModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(1, 1), alpha=0.001, random_state=42),
                    features_and_labels, foo='bar'),
            SkModel(LogisticRegression(), features_and_labels),
            SkModel(LinearSVC(), features_and_labels),
            SkModel(RandomForestClassifier(), features_and_labels),
            KerasModel(keras_model_provider, features_and_labels),
            MultiModel(SkModel(LogisticRegression(), FeaturesAndLabels(["a"], {"b": ["b"]})))
        ]

        """when"""
        fits = [df.fit(mp) for mp in providers]
        models = []
        for i, f in enumerate(fits):
            f.save_model(f'/tmp/pandas-ml-utils-unittest-test_model_{i}')
            models.append((f.model, Model.load(f'/tmp/pandas-ml-utils-unittest-test_model_{i}')))

        """then"""
        for i, (fitted_model, restored_model) in enumerate(models):
            print(f"test model ==> {i}")
            pd.testing.assert_frame_equal(df.predict(fitted_model), df.predict(restored_model))
            pd.testing.assert_frame_equal(df.backtest(fitted_model).df, df.backtest(restored_model).df)

    def test_model_with_LazyDataFrame_copy(self):
        """given"""
        model = SkModel(
            MLPClassifier(activation='tanh', hidden_layer_sizes=(1, 1), alpha=0.001, random_state=42),
            FeaturesAndLabels([], []), foo='bar', ldf=LazyDataFrame(None, foo=lambda _f: 'bar'))

        """when"""
        model.save(f'/tmp/pandas-ml-utils-unittest-test_model_LDF')
        model2 = Model.load(f'/tmp/pandas-ml-utils-unittest-test_model_LDF')

        """then"""
        self.assertEqual(model.kwargs["ldf"], model2.kwargs["ldf"])
        self.assertEqual(model.kwargs["ldf"].kwargs['foo'](None), 'bar')
        self.assertEqual(model2.kwargs["ldf"].kwargs['foo'](None), 'bar')

    def test_save_load_keras_custom_loss(self):
        """given"""
        features_and_labels = FeaturesAndLabels(["a"], ["b"])
        name = '/tmp/pandas-ml-utils-unittest-test_model_keras_custom_loss'

        def loss_provider(foo):
            def my_custom_loss(x, y):
                print(foo)
                import keras.backend as K
                return K.sum(x - y)

            return my_custom_loss

        def keras_model_provider():
            model = Sequential()
            model.add(Dense(1, input_dim=1, activation='sigmoid'))

            model.compile(optimizer='Adam', loss=loss_provider("bar"))
            return model, loss_provider("bar")

        """when"""
        fit = df.fit(KerasModel(keras_model_provider, features_and_labels, optimizer='adam', verbose=0))
        fitted_model = fit.model

        fit.save_model(name)
        restored_model = Model.load(name)

        """then"""
        pd.testing.assert_frame_equal(df.predict(fitted_model), df.predict(restored_model))
        pd.testing.assert_frame_equal(df.backtest(fitted_model).df, df.backtest(restored_model).df)

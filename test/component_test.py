import pandas as pd
import numpy as np
import unittest

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy
from sklearn.model_selection import KFold

import pandas_ml_utils as pdu

from sklearn.neural_network import MLPClassifier, MLPRegressor

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
                                              pdu.FeaturesAndLabels(features=['vix_Close'], labels=['label'],
                                                                    target_columns=["vix_Open"],
                                                                    loss_column="spy_Volume")),
                                test_size=0.4,
                                test_validate_split_seed=42)

        self.assertEqual(fit.model.min_required_data, 1)
        np.testing.assert_array_equal(fit.test_summary.confusion_count(), np.array([[744, 586],
                                                                                    [655, 698]]))

        # backtest
        fitted_model = fit.model
        backtest_classification = df.backtest_classifier(fitted_model)
        np.testing.assert_array_equal(backtest_classification.confusion_count(), np.array([[1811, 1458],
                                                                                           [1657, 1780]]))

        # classify
        fitted_model = fit.model
        classified_df = df.classify(fitted_model)
        print(classified_df.tail())

        self.assertEqual(len(classified_df[classified_df["prediction"] == False]), 3437)
        self.assertTrue(classified_df["loss"].sum() > 0)
        self.assertTrue(classified_df["prediction_proba"].sum() > 0)
        self.assertTrue(classified_df["prediction_proba"].min() > 0)
        self.assertTrue(classified_df["prediction_proba"].max() < 1)
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

    def test_fit_regressor(self):
        df = pd.read_csv(f'{__name__}.csv', index_col='Date') / 50.

        # fit
        fit = df.fit_regressor(pdu.SkitModel(MLPRegressor(activation='tanh', hidden_layer_sizes=(4, 3, 2, 1, 2, 3, 4),
                                                          random_state=42),
                                             pdu.FeaturesAndLabels(
                                                 features=['vix_Open', 'vix_High', 'vix_Low', 'vix_Close'],
                                                 labels=['vix_Open', 'vix_High', 'vix_Low', 'vix_Close'])),
                               test_size=0.4,
                               test_validate_split_seed=42)

        fitted_model = fit.model

        # backtest
        backtest_regression = df.backtest_regressor(fitted_model)
        self.assertIsNotNone(backtest_regression) # FIXME better assertion

        # regressed
        regressed = df.regress(fitted_model)
        print(regressed.tail())
        self.assertListEqual(regressed.columns.tolist(),
                             ['vix_Open', 'vix_High', 'vix_Low', 'vix_Close',
                              'prediction_vix_Open', 'prediction_vix_High', 'prediction_vix_Low', 'prediction_vix_Close',
                              'error'])

        self.assertTrue(regressed["error"].sum() > 0)
        self.assertTrue(regressed["error"].sum() > 0)
        self.assertTrue(regressed["error"].min() > 0)

    def test_cross_validation(self):
        df = pd.read_csv(f'{__name__}.csv', index_col='Date')
        df['label'] = df["spy_Close"] > df["spy_Open"]

        # KFold
        cv = KFold(n_splits=10, shuffle=False)

        # fit
        fit = df.fit_classifier(pdu.SkitModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), alpha=0.001,
                                                            random_state=42, max_iter=10),
                                              pdu.FeaturesAndLabels(features=['vix_Close'],
                                                                    labels=['label'],
                                                                    target_columns=["vix_Open"],
                                                                    loss_column="spy_Volume")),
                                test_size=0.4,
                                cross_validation = (2, cv.split),
                                test_validate_split_seed=42)

        self.assertEqual(fit.model.min_required_data, 1)
        np.testing.assert_array_equal(fit.test_summary.confusion_count(), np.array([[257, 169],
                                                                                    [1142, 1115]]))

    def test_reinforcement_model(self):
        df = pd.read_csv(f'{__name__}.csv', index_col='Date')
        df['vix_Close'] = df['vix_Close'] / 50
        df['label'] = (df["spy_Close"] - df["spy_Open"]).shift(-1)
        df = df.drop('2019-09-13').dropna()

        # define agent with model:
        def agent_provider(observation_space_shape, nb_actions):
            # define model
            model = Sequential()
            model.add(Flatten(input_shape=(1,) + observation_space_shape))
            model.add(Dense(16))
            model.add(Activation('relu'))
            model.add(Dense(nb_actions))
            model.add(Activation('relu'))

            # define agent
            policy = BoltzmannQPolicy()
            sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
            sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

            return sarsa

        # fit
        fit = df.fit_agent(pdu.OpenAiGymModel(agent_provider,
                                              pdu.FeaturesAndLabels(features=['vix_Close'],
                                                                    labels=['label'],
                                                                    label_type=float,
                                                                    target_columns=["vix_Close", "spy_Close"],
                                                                    loss_column="spy_Volume"),
                                              [
                                                  lambda y: 0, # do nothing
                                                  lambda y: y  # get trade reward
                                              ],
                                              [df['label'].min(), df['label'].max()],
                                              episodes=2),
                           test_size=0.4,
                           test_validate_split_seed=42)

        print(fit.training_summary.get_data_frame().tail())
        print(fit.test_summary.get_data_frame().tail())
        self.assertTrue(True)

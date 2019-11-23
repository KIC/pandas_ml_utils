import logging
import unittest

import numpy as np
import pandas as pd
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents import SARSAAgent
from rl.policy import MaxBoltzmannQPolicy
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier, MLPRegressor

import pandas_ml_utils as pdu
from pandas_ml_utils.analysis.correlation_analysis import _sort_correlation

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class ComponentTest(unittest.TestCase):

    def test_fetch_yahoo(self):
        df = pd.fetch_yahoo(spy="SPY").tail()
        self.assertTrue(df["spy_Close"].sum() > 0)

    def test_correlation_matrix(self):
        df = pd.read_csv(f'{__name__}.csv', index_col='Date')[["spy_Open", "spy_High", "spy_Low", "spy_Close"]]
        corr = _sort_correlation(df.corr(), recursive=True)

        self.assertListEqual(corr.columns.tolist(), ["spy_Open", "spy_High", "spy_Close", "spy_Low"])

    def test_make_train_data(self):
        df = pd.read_csv(f'{__name__}.csv', index_col='Date')
        df['label'] = df["spy_Close"] > df["spy_Open"]

        x_train, x_test, y_train, y_test, index_train, index_test = \
            df.make_training_data(pdu.FeaturesAndLabels(features=['vix_Close'], labels=['label'], feature_lags=[0, 1, 2]))

        self.assertEqual(x_train.shape, (4022, 3, 1))
        self.assertEqual(y_train.shape, (4022, 1))

        self.assertEqual(x_test.shape, (2682, 3, 1))
        self.assertEqual(y_test.shape, (2682, 1))

        self.assertEqual(len(x_train), len(index_train))
        self.assertEqual(len(x_test), len(index_test))

    def test_fit_classifier_full(self):
        df = pd.read_csv(f'{__name__}.csv', index_col='Date')
        df['label'] = df["spy_Close"] > df["spy_Open"]

        # fit
        fit = df.fit_classifier(pdu.SkitModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), alpha=0.001,
                                                            random_state=42),
                                              pdu.FeaturesAndLabels(features=['vix_Close'], labels=['label'],
                                                                    targets=("vix_Open", "spy_Volume"))),
                                test_size=0.4,
                                test_validate_split_seed=42)

        self.assertEqual(fit.model.features_and_labels.min_required_samples, 1)
        np.testing.assert_array_equal(fit.training_summary.get_confusion_matrix()['vix_Open'], np.array([[1067,  872], [1002, 1082]]))
        np.testing.assert_array_equal(fit.test_summary.get_confusion_matrix()['vix_Open'], np.array([[744, 586], [655, 698]]))

        # backtest
        fitted_model = fit.model
        backtest_classification = df.backtest_classifier(fitted_model)
        np.testing.assert_array_equal(backtest_classification.get_confusion_matrix()['vix_Open'], np.array([[1811, 1458], [1657, 1780]]))

        # classify
        fitted_model = fit.model
        classified_df = df.classify(fitted_model)
        print(classified_df.tail())

        self.assertEqual(len(classified_df[classified_df["vix_Open", "prediction", "value"] == False]), 3437)
        self.assertTrue(classified_df["vix_Open", "prediction", "value_proba"].sum() > 0)
        self.assertTrue(classified_df["vix_Open", "prediction", "value_proba"].min() > 0)
        self.assertTrue(classified_df["vix_Open", "prediction", "value_proba"].max() < 1)
        self.assertListEqual(classified_df.columns.tolist(),
                             [('vix_Open', 'prediction', 'value'), ('vix_Open', 'prediction', 'value_proba'), ('vix_Open', 'target', 'value')])

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
                             [('target', 'prediction', 'value'), ('target', 'prediction', 'value_proba'), ('target', 'target', 'value')])

    def test_fit_classifier_tail(self):
        import talib
        df = pd.read_csv(f'{__name__}.csv', index_col='Date')
        df['label'] = df["spy_Close"] > df["spy_Open"]

        # fit
        fit = df.fit_classifier(pdu.SkitModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), alpha=0.001,
                                                            random_state=42),
                                              pdu.FeaturesAndLabels(['vix_Close'], ['label'],
                                                                    feature_lags=[0, 1],
                                                                    lag_smoothing={1: lambda df: talib.SMA(df["vix_Close"], timeperiod=2)})),
                                test_size=0.4,
                                test_validate_split_seed=42)

        fitted_model = fit.model
        classified_df = df.classify(fitted_model, tail=1)

        self.assertEqual(len(classified_df), 1)
        self.assertTrue(classified_df[('target', 'prediction', 'value_proba')][-1:].values[0] > 0.0)

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
        self.assertIsNotNone(backtest_regression)

        # regressed
        regressed = df.regress(fitted_model)
        print(regressed.tail())
        self.assertListEqual(regressed.columns.tolist(),
                             [('target', 'target', 'value'),
                              ('target', 'prediction', 'vix_Open'),
                              ('target', 'prediction', 'vix_High'),
                              ('target', 'prediction', 'vix_Low'),
                              ('target', 'prediction', 'vix_Close')])

        self.assertEqual(len(regressed), 6706)

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
                                                                    targets=("vix_Open", "spy_Volume"))),
                                test_size=0.4,
                                cross_validation = (2, cv.split),
                                test_validate_split_seed=42)

        self.assertEqual(fit.model.features_and_labels.min_required_samples, 1)
        np.testing.assert_array_equal(fit.test_summary.get_confusion_matrix()["vix_Open"],
                                      np.array([[257, 169], [1142, 1115]]))

    def test_hyper_parameter(self):
        from hyperopt import hp

        df = pd.read_csv(f'{__name__}.csv', index_col='Date')
        df['label'] = df["spy_Close"] > df["spy_Open"]

        # fit with find hyper parameter
        fit = df.fit_classifier(
            pdu.SkitModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
                          pdu.FeaturesAndLabels(features=['vix_Close'], labels=['label'],
                                                target_columns=["vix_Open"],
                                                loss_column="spy_Volume")),
            test_size=0.4,
            test_validate_split_seed=42,
            hyper_parameter_space={'alpha': hp.choice('alpha', [0.0001, 10]), 'early_stopping': True, 'max_iter': 50,
                                   '__max_evals': 4, '__rstate': np.random.RandomState(42)}
        )

        # test best parameter
        self.assertEqual(fit.model.skit_model.get_params()['alpha'], 0.0001)

    def test_multi_model(self):
        df = pd.read_csv(f'{__name__}.csv', index_col='Date')
        df['label1'] = df["spy_Close"] > df["spy_Open"]
        df['label2'] = df["spy_Close"] > df["spy_Open"]

        # define the model
        model = pdu.SkitModel(MLPRegressor(activation='tanh', hidden_layer_sizes=(60, 50), alpha=0.001, random_state=42),
                              pdu.FeaturesAndLabels(features=['vix_Close'], labels=['vix_High', 'vix_Low'],
                                                    targets={"vix_High": ("spy_Volume", "vix_High"),
                                                             "vix_Low": ("spy_Volume", "vix_Low")}))
        # fit
        fit = df.fit_regressor(pdu.MultiModel(model),
                               test_size=0.4,
                               test_validate_split_seed=42)

        self.assertListEqual(fit.training_summary.df.columns.tolist(), fit.test_summary.df.columns.tolist())
        self.assertListEqual(fit.training_summary.df.columns.tolist(), [('vix_High', 'target', 'value'),
                                                                        ('vix_High', 'prediction', 'value'),
                                                                        ('vix_Low', 'target', 'value'),
                                                                        ('vix_Low', 'prediction', 'value'),
                                                                        ('vix_High', 'label', 'value'),
                                                                        ('vix_Low', 'label', 'value'),
                                                                        ('vix_High', 'loss', 'value'),
                                                                        ('vix_Low', 'loss', 'value')])

    @unittest.skip("we need a better model for reinforcement learning")
    def test_reinforcement_model(self):
        df = pd.read_csv(f'{__name__}.csv', index_col='Date')
        df['vix_Close'] = df['vix_Close'] / 50
        df['label'] = (df["spy_Close"] - df["spy_Open"]).shift(-1)
        df = df.drop('2019-09-13').dropna()[-100:]

        # define agent with model:
        def agent_provider(observation_space_shape, nb_actions):
            print(f"observation_space_shape: {observation_space_shape}")

            # define model
            model = Sequential()
            model.add(Flatten(input_shape=(1,) + observation_space_shape))
            model.add(Dense(16))
            model.add(Activation('sigmoid'))
            model.add(Dense(nb_actions))
            model.add(Activation('relu'))

            # define agent
            policy = MaxBoltzmannQPolicy()
            sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=0, policy=policy)
            sarsa.compile(Adam(lr=1e-4), metrics=['mae'])

            return sarsa

        # fit
        fit = df.fit_agent(pdu.OpenAiGymModel(agent_provider,
                                              pdu.FeaturesAndLabels(features=['vix_Close'],
                                                                    labels=['label'],
                                                                    feature_lags=range(5),
                                                                    label_type=float,
                                                                    target_columns=["vix_Close", "spy_Close"],
                                                                    loss_column="spy_Volume"),
                                              [
                                                  lambda y: -0.02, # do nothing
                                                  lambda y: y  # get trade reward
                                              ],
                                              [df['label'].min(), df['label'].max()],
                                              episodes=15),
                           test_size=0.4,
                           test_validate_split_seed=42)

        print(fit.training_summary.get_data_frame().tail())
        print(fit.training_summary.get_data_frame()["reward_history"].sum())
        print(fit.test_summary.get_data_frame().tail())
        print(fit.test_summary.get_data_frame()["reward_history"].sum())

        self.assertTrue(fit.training_summary.get_data_frame()["action_history"].sum() > 1)
        self.assertTrue(fit.test_summary.get_data_frame()["action_history"].sum() > 1)

from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC

from pandas_ml_utils.model.fitting.fitter import fit, backtest, predict
from pandas_ml_utils.model.models import *
from pandas_ml_utils.constants import *

df = pd.DataFrame({"a": np.array([0.1, 0.01]), "b": np.array([True, False]), "c": np.array([False, True])})


class TestFitter(TestCase):

    def test__fit(self):
        """given"""
        features_and_labels = FeaturesAndLabels(["a"], ["b"], targets=lambda f: f["b"])
        providers = [
            SkModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(1, 1), alpha=0.001, random_state=42),
                    features_and_labels, foo='bar'),
            SkModel(LogisticRegression(), features_and_labels),
            SkModel(LinearSVC(), features_and_labels),
            SkModel(RandomForestClassifier(), features_and_labels)]

        """when"""
        fits = [fit(df, p, 0) for p in providers]
        summaries = [f.training_summary.df for f in fits]
        fits_df_columns = [f.columns.tolist() for f in summaries]

        """then"""
        expected_columns = [(PREDICTION_COLUMN_NAME, 'b'), (LABEL_COLUMN_NAME, 'b'), (TARGET_COLUMN_NAME, "b")]
        self.assertListEqual(fits_df_columns[0], expected_columns)
        self.assertListEqual(fits_df_columns[1], expected_columns)
        self.assertListEqual(fits_df_columns[2], expected_columns)
        self.assertListEqual(fits_df_columns[3], expected_columns)
        np.testing.assert_array_equal(summaries[0][TARGET_COLUMN_NAME, "b"].values, df["b"].values)
        np.testing.assert_array_equal(summaries[1][TARGET_COLUMN_NAME, "b"].values, df["b"].values)
        np.testing.assert_array_equal(summaries[2][TARGET_COLUMN_NAME, "b"].values, df["b"].values)
        np.testing.assert_array_equal(summaries[3][TARGET_COLUMN_NAME, "b"].values, df["b"].values)

        """and has html representation"""
        self.assertTrue(len(fits[0]._repr_html_()) >= 2095)

    def test__backtest(self):
        """given"""
        fls = [FeaturesAndLabels(["a"], ["b"]),
               FeaturesAndLabels(["a"], ["b"], targets=lambda f: f["b"]),
               FeaturesAndLabels(["a"], ["b", "c"], targets=lambda f: -1),
               FeaturesAndLabels(["a"], {"b": ["b"], "a": ["c"]},
                                 targets=lambda f, t: pd.Series(-1 if t == "b" else -2, index=f.index, name=t),
                                 gross_loss=lambda f, t: pd.Series(-1, index=f.index, name=t))
               ]

        providers = [SkModel(MLPRegressor(activation='tanh', hidden_layer_sizes=(1, 1), alpha=0.001, random_state=42),
                             features_and_labels=fl) for fl in fls]

        """when"""
        fitted_models = [fit(df, p, 0).model for p in providers]
        backtests = [backtest(df, fm) for fm in fitted_models]
        backtest_columns = [b.df.columns.tolist() for b in backtests]

        """then"""
        source_columns = (SOURCE_COLUMN_NAME, 'a'), (SOURCE_COLUMN_NAME, 'b'), (SOURCE_COLUMN_NAME, 'c')
        self.assertEqual(backtest_columns[0], [(PREDICTION_COLUMN_NAME, 'b'), (LABEL_COLUMN_NAME, 'b'), *source_columns])
        self.assertEqual(backtest_columns[1], [(PREDICTION_COLUMN_NAME, 'b'), (LABEL_COLUMN_NAME, 'b'), (TARGET_COLUMN_NAME, 'b'), *source_columns])
        self.assertEqual(backtest_columns[2], [(PREDICTION_COLUMN_NAME, 'b'), (PREDICTION_COLUMN_NAME, 'c'), (LABEL_COLUMN_NAME, 'b'), (LABEL_COLUMN_NAME, 'c'), (TARGET_COLUMN_NAME, TARGET_COLUMN_NAME), *source_columns])
        self.assertEqual(backtest_columns[3], [('b', PREDICTION_COLUMN_NAME, 'b'), ('a', PREDICTION_COLUMN_NAME, 'c'), ('b', LABEL_COLUMN_NAME, 'b'), ('a', LABEL_COLUMN_NAME, 'c'), ('b', GROSS_LOSS_COLUMN_NAME, 'b'), ('a', GROSS_LOSS_COLUMN_NAME, 'a'), ('b', TARGET_COLUMN_NAME, 'b'), ('a', TARGET_COLUMN_NAME, 'a'), *source_columns])
        np.testing.assert_array_almost_equal(backtests[3].df["b", PREDICTION_COLUMN_NAME, "b"].values, np.array([-1.51, -1.52]), 2)
        np.testing.assert_array_almost_equal(backtests[3].df["a", PREDICTION_COLUMN_NAME, "c"].values, np.array([0.56, 0.56]), 2)
        np.testing.assert_array_equal(backtests[3].df["b", LABEL_COLUMN_NAME, "b"].values, df["b"].values)
        np.testing.assert_array_equal(backtests[3].df["a", LABEL_COLUMN_NAME, "c"].values, df["c"].values)
        np.testing.assert_array_equal(backtests[3].df["a", GROSS_LOSS_COLUMN_NAME, "a"].values, -1)
        np.testing.assert_array_equal(backtests[3].df["b", GROSS_LOSS_COLUMN_NAME, "b"].values, -1)
        np.testing.assert_array_equal(backtests[3].df["a", TARGET_COLUMN_NAME, "a"].values, -2)
        np.testing.assert_array_equal(backtests[3].df["b", TARGET_COLUMN_NAME, "b"].values, -1)

    def test__predict(self):
        """given"""
        fls = [FeaturesAndLabels(["a"], ["b"]),
               FeaturesAndLabels(["a"], ["b"], targets=lambda f: f["b"]),
               FeaturesAndLabels(["a"], ["b", "c"], targets=lambda f: -1),
               FeaturesAndLabels(["a"], {"b": ["b"], "a": ["c"]},
                                 targets=lambda f, t: pd.Series(-1 if t == "b" else -2, index=f.index, name=t),
                                 gross_loss=lambda f, t: pd.Series(-1, index=f.index, name=t))
               ]

        providers = [SkModel(MLPRegressor(activation='tanh', hidden_layer_sizes=(1, 1), alpha=0.001, random_state=42),
                             features_and_labels=fl) for fl in fls]


        """when"""
        fitted_models = [fit(df, p, 0).model for p in providers]

        """then"""
        predictions = [predict(df, fm) for fm in fitted_models]
        print(predictions[-1].columns.tolist())
        self.assertEqual(predictions[0].columns.tolist(), [(PREDICTION_COLUMN_NAME, 'b')])
        self.assertEqual(predictions[1].columns.tolist(), [(PREDICTION_COLUMN_NAME, 'b'), (TARGET_COLUMN_NAME, 'b')])
        self.assertEqual(predictions[2].columns.tolist(), [(PREDICTION_COLUMN_NAME, 'b'), (PREDICTION_COLUMN_NAME, 'c'), (TARGET_COLUMN_NAME, TARGET_COLUMN_NAME)])
        self.assertEqual(predictions[3].columns.tolist(), [('b', PREDICTION_COLUMN_NAME, 'b'), ('a', PREDICTION_COLUMN_NAME, 'c'), ('b', TARGET_COLUMN_NAME, 'b'), ('a', TARGET_COLUMN_NAME, 'a')])

    def test__predict_with_lags(self):
        """given"""
        df = pd.DataFrame({"a": [0.5592344, 0.60739384, 0.19994533, 0.56642537, 0.50965677,
                                 0.168989, 0.94080671, 0.76651769, 0.8403563, 0.4003567,
                                 0.24295908, 0.50706317, 0.66612371, 0.4020924, 0.21776017,
                                 0.32559497, 0.12721287, 0.13904584, 0.65887554, 0.08830925],
                           "b": range(20)})

        fl = FeaturesAndLabels(["a"], ["b"], feature_lags=[0,1,2])
        provider = SkModel(MLPRegressor(activation='tanh', hidden_layer_sizes=(1, 1), alpha=0.001, random_state=42),
                           features_and_labels=fl)

        """when"""
        fitted = fit(df, provider, 0)
        predictions = predict(df, fitted.model)

        """then"""
        self.assertListEqual(predictions.columns.tolist(), [(PREDICTION_COLUMN_NAME, 'b')])
        self.assertEqual(fitted.model.features_and_labels.min_required_samples, 3)

import os
from unittest import TestCase

from keras.layers import Dense
from keras.models import Sequential
from test.config import TEST_FILE

from pandas_ml_utils import pd, KerasModel, FeaturesAndLabels

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestKerasLossWeight(TestCase):

    def test_single_model(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["feature"] = df["spy_Close"].pct_change()
        df["label"] = df["spy_Close"] > df["spy_Open"]
        df["weight"] = df["spy_Close"].pct_change().abs()

        def model_provider():
            model = Sequential()
            model.add(Dense(1, input_shape=(1,)))
            model.add(Dense(10))
            model.add(Dense(1))

            model.compile(
                loss="mse",
                optimizer='adam',
                metrics=['accuracy'],
            )

            return model

        model = KerasModel(
            model_provider,
            FeaturesAndLabels(
                features=["feature"],
                labels=["label"],
                weights="weight"
            )
        )

        """when"""
        fit = df.fit(model)

        """then"""
        # no exception is thrown


    def test_multi_model(self):
        pass


import logging
import unittest
from typing import List

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

import pandas_ml_utils as pdu
from pandas_ml_utils.constants import *
from test.config import TEST_FILE

from pandas_ml_utils.model.features_and_labels.target_encoder import TargetLabelEncoder
from test.mocks.mock_model import MockModel

from pandas_ml_utils.utils.functions import integrate_nested_arrays

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class EncoderTest(unittest.TestCase):

    def test__2d_encoding(self):
        """given"""
        df = pd.read_csv(TEST_FILE, index_col='Date')
        df["label"] = df["spy_Close"] > df["spy_Open"]

        class ArrayEncoder(TargetLabelEncoder):

            def __init__(self):
                super().__init__()

            @property
            def labels_source_columns(self) -> List[str]:
                return ["spy_Close"]

            @property
            def encoded_labels_columns(self) -> List[str]:
                return ["2D"]

            def encode(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
                res = pd.DataFrame({}, index=df.index)
                res["2D"] = df["spy_Close"] = df["spy_Close"].apply(lambda r: np.array([r, r]))
                return res

        """when"""
        model = MockModel(pdu.FeaturesAndLabels(["spy_Close"], ArrayEncoder(), feature_lags=[0, 1, 2]))
        fit = df.fit(model)

        """then"""
        print(fit.test_summary.df)
        self.assertEqual(fit.test_summary.df.shape, (2682, 2))
        self.assertEqual(integrate_nested_arrays(fit.test_summary.df.values).shape, (2682, 2, 2))

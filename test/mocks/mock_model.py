import numpy as np

from pandas_ml_utils import Model


class MockModel(Model):

    def __init__(self, fnl):
        super().__init__(fnl)
        self.label_shape = None

    def fit(self, x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, sample_weight_train: np.ndarray,
            sample_weight_test: np.ndarray) -> float:
        self.label_shape = y.shape
        return 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.ones((len(x), *self.label_shape[1:]))
import numpy as np
from typing import List, Tuple


class TrainTestData():

    def __init__(self,
                 x_train: np.ndarray,
                 x_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 names: Tuple[List[str], List[str]]) -> None:
        self.features = names[0]
        self.labels = names[1]
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def values(self):
        return self.x_train, self.x_test, self.y_train, self.y_test, (self.features, self.labels)
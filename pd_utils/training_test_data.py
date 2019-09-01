import numpy as np
from typing import List, Tuple, Callable, Any


class TrainTestData(object):

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

    def fit_sklearn_classifier(self, sklearn_model, probability_cutoff=0.5):
        return self.fit_classifier(lambda: sklearn_model,
                                   lambda model, x, y: model.fit(x, y),
                                   lambda model, x: model.predict_proba(x)[:,1],
                                   probability_cutoff)

    def fit_classifier(self,
                       model_provider: Callable[[], Any],
                       model_fitter: Callable[[Any, np.ndarray, np.ndarray], Any],
                       model_predictor: Callable[[Any, np.ndarray], np.ndarray],
                       probability_cutoff: float = 0.5):

        from sklearn.metrics import confusion_matrix
        model = model_provider()
        res = model_fitter(model, self.x_train, self.y_train)

        if isinstance(res, type(model)):
            model = res

        train_confusion = confusion_matrix(self.y_train, model_predictor(model, self.x_train) > probability_cutoff)
        test_confusion = confusion_matrix(self.y_test, model_predictor(model, self.x_test) > probability_cutoff)

        return model, train_confusion, test_confusion

    def with_rnn_as_ar_shape(self) -> 'TrainTestData':
        if len(self.x_train.shape) < 3:
            print("Data was not in RNN shape")
            return self
        else:
            return TrainTestData(self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1] * self.x_train.shape[2]),
                                 self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1] * self.x_test.shape[2]),
                                 self.y_train,
                                 self.y_test,
                                 (self.features.reshape(self.x_test.shape[1] * self.x_test.shape[2]), self.labels))

    def values(self):
        return self.x_train, self.x_test, self.y_train, self.y_test, (self.features, self.labels)
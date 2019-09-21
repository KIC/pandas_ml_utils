import logging
from collections.abc import Iterable
from itertools import groupby

from typing import Tuple, Dict, Callable, Any, Union, List
from .utils import unfold_parameter_space
import pandas_ml_utils as pdu
import dill as pickle
import pandas as pd
import numpy as np


log = logging.getLogger(__name__)


class MultiModel(object):

    @staticmethod
    def load(filename: str):
        with open(filename, 'rb') as file:
            multi_model = pickle.load(file)
            if isinstance(multi_model, MultiModel):
                return multi_model
            else:
                raise ValueError("file provided was not a MultiModel")

    def __init__(self,
                 data_provider: Callable[[], pd.DataFrame],
                 data_engineer: Callable[[pd.DataFrame], pd.DataFrame],
                 model_provider: Callable[[], pdu.Model],
                 parameter_space: Dict[str, Iterable]):
        self.data_provider = data_provider
        self.data_engineer = data_engineer
        self.model_provider = model_provider
        self.parameter_space = unfold_parameter_space(parameter_space.copy(), {})
        self.min_needed_data: int = None
        self.data: pd.DataFrame = None
        self.fits: List[pdu.Fit] = None

    def save(self, filename: str):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    # def fetch_data_and_fit(self, test_size: float = 0.4, test_validate_split_seed: int=None):
    #     self.fetch_data()
    #     self.fit()

    def fetch_data(self):
        self.data = self.data_provider()

    def fit(self, test_size: float = 0.4, test_validate_split_seed: int = None) -> None:
        def model_fitter(**kwargs) -> pdu.Fit:
            fit = self.data_engineer(self.data, **kwargs) \
                      .fit_classifier(self.model_provider,
                                      test_size=test_size,
                                      test_validate_split_seed=test_validate_split_seed)

            log.info(f'fit for { {**kwargs}}\n{fit.training_classification.confusion_count()}\n{fit.test_classification.confusion_count()}')
            return fit

        # TODO there should be a way to generate one ClassificationSummary out of several by summing or averaging
        self.fits = [model_fitter(**kwargs) for kwargs in self.parameter_space]
        self.min_needed_data = max([fit.model.min_required_data for fit in self.fits])

    def predict(self) -> pd.DataFrame:
        df = self.data[-self.min_needed_data:] if self.min_needed_data is not None else self.data

        def model_predictor(model, **kwargs) -> pd.DataFrame:
            prediction = self.data_engineer(df, **kwargs) \
                             .classify(model)

            return prediction[-1:]

        predictions = [model_predictor(self.fits[i].model, **kwargs) for i, kwargs in enumerate(self.parameter_space)]
        return predictions

    def plot_heatmap(self, parameter_as_column: str):
        import seaborn as sns
        sns.heatmap(self.compute_heatmap(parameter_as_column))

    def compute_heatmap(self, parameter_as_column: str):
        predictions = self.predict()

        # to group all ro indices per column index we first need to sort accordingly
        sorted_parameter_space = sorted(enumerate(self.parameter_space), key=lambda x: x[1][parameter_as_column])

        columns = {col: [value[0] for value in parameter]
                   for col, parameter in groupby(sorted_parameter_space, lambda x: x[1][parameter_as_column])}

        # assign a data frame for each column
        predictions = [pd.concat([predictions[row][["target", "prediction_proba"]] for row in rows], axis=0, sort=True) \
                         .set_index("target") \
                         .groupby(level=0).max() \
                         .rename(columns={"prediction_proba": column})
                       for column, rows in columns.items()]

        predictions = pd.concat(predictions, axis=1, sort=True).sort_index(ascending=False)
        return predictions

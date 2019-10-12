import logging
from typing import Callable, Tuple

import numpy as np
import pandas as pd

from ..model.fit import Fit
from ..model.fitter import _fit
from ..model.models import Model
from .summary import ReinforcementSummary

log = logging.getLogger(__name__)


def fit_agent(df: pd.DataFrame,
              model_provider: Callable[[int], Model],
              test_size: float = 0.4,
              cross_validation: Tuple[int, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
              cache_feature_matrix: bool = False,
              test_validate_split_seed = 42,
              summary_printer: Callable[[np.ndarray, np.ndarray, np.ndarray], None] = None
              ) -> Fit:

    model, train, test, index = _fit(df,
                                     model_provider,
                                     test_size = test_size,
                                     cross_validation = cross_validation,
                                     cache_feature_matrix = cache_feature_matrix,
                                     test_validate_split_seed = test_validate_split_seed,
                                     summary_printer = summary_printer)

    train_targets = df[model.features_and_labels.target_columns].loc[index[0]]
    test_targets = df[model.features_and_labels.target_columns].loc[index[1]]

    training_classification = ReinforcementSummary(train_targets, model.history[0])
    test_classification = ReinforcementSummary(test_targets, model.history[1])
    return Fit(model, training_classification, test_classification)

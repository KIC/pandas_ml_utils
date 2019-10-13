import logging
import sys
from typing import Tuple, Union

import numpy as np
import pandas as pd

from ..model.summary import Summary

log = logging.getLogger(__name__)


class ReinforcementSummary(Summary):

    def __init__(self, target: pd.DataFrame, agent_history: pd.DataFrame):
        self.df = target.join(agent_history).sort_index()

    def get_data_frame(self):
        return self.df

    def _repr_html_(self):
        return self._html_()._repr_html_()

    def _html_(self, width: str = '100%'):
        # only import it needed
        from vdom.helpers import div, p, img, table, tr, td, tbody, thead, th
        import matplotlib.pyplot as plt
        import base64
        import io

        return div(p("TODO"))

import pandas as pd
from typing import Dict


class Summary(object):

    def __init__(self, target_data: Dict[str, pd.DataFrame], **kwargs):
        self.target_data = target_data
        self.kwargs = kwargs

    def _repr_html_(self):
        pass

    def _html_(self, width: str = '100%'):
        pass

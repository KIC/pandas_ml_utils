from typing import Any

import pandas as pd

from .models import Model
from .summary import Summary


class Fit(object):

    def __init__(self,
                 model: Model,
                 training_summary: Summary,
                 test_summary: Summary,
                 trails: Any):
        self.model = model
        self.training_summary = training_summary
        self.test_summary = test_summary
        self._trails = trails

    def values(self):
        return self.model, self.training_summary, self.test_summary

    def trails(self):
        if self._trails is not None:
            return pd.DataFrame(self._trails.results)\
                     .drop("parameter", axis=1)\
                     .join(pd.DataFrame([r['parameter'] for r in self._trails.results]))
        else:
            return None

    def _repr_html_(self):
        from mako.template import Template
        from mako.lookup import TemplateLookup

        template = Template(filename=f"{__file__}.html", lookup=TemplateLookup(directories=['/']))
        return template.render(fit=self)
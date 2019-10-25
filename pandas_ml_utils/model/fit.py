import pandas as pd
from typing import Any

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
        return self._html_()._repr_html_()

    def _html_(self):
        # only import it needed
        from vdom.helpers import div, table, tr, td, tbody, thead, th

        model = self.model.__repr__()
        if model is None:
            model = str(self.model)

        return div(
            table(
                thead(
                    tr(
                        th("Training Data", style={'text-align': 'left'}),
                        th("Test Data", style={'text-align': 'right'})
                    )
                ),
                tbody(
                    tr(
                        td(self.training_summary._html_()),
                        td(self.test_summary._html_())
                    ),
                    tr(
                        td(
                            model,
                            colspan="2"
                        )
                    )
                ),
                style={'width': '100%'}
            ),
            style={'width': '100%', 'float': 'left'}
        )
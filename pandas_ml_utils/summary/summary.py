import pandas as pd
import os


class Summary(object):

    def __init__(self, df: pd.DataFrame, **kwargs):
        self._df = df
        self.kwargs = kwargs

    @property
    def df(self):
        return self._df

    def _html_template_file(self):
        return f"{os.path.abspath(__file__)}.html"

    def _repr_html_(self):
        from mako.template import Template
        from mako.lookup import TemplateLookup

        template = Template(filename=self._html_template_file(), lookup=TemplateLookup(directories=['/']))
        return template.render(summary=self)


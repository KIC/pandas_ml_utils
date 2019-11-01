import logging
import sys
from functools import lru_cache
from typing import Tuple, Union, Dict

import numpy as np
import pandas as pd

from ..constants import *
from ..model.summary import Summary
from sklearn.metrics import f1_score

log = logging.getLogger(__name__)


class ClassificationSummary(Summary):
    def __init__(self, df: pd.DataFrame, probability_cutoff: float = 0.5):
        super().__init__()
        self.df = df
        self.probability_cutoff = probability_cutoff
        self.confusions = {target: ClassificationSummary._calculate_confusions(df[target]) for target in {*df.columns.get_level_values(0)}}

    @lru_cache(maxsize=None)
    def get_confusion_matrix(self):
        return {target: np.array([[len(c) for c in r] for r in cm]) for target, cm in self.confusions.items()}

    @lru_cache(maxsize=None)
    def get_confusion_loss(self):
        return {target: np.array([[c[LOSS_COLUMN_NAME, "value"].sum() for c in r] for r in cm]) for target, cm in self.confusions.items()}

    @lru_cache(maxsize=None)
    def get_metrics(self):
        return {target: {"FP Ratio": fp_ratio,
                         "FN Ratio": fn_ratio,
                         "F1 Score": f1_score(self.df[target, LABEL_COLUMN_NAME, "value"],
                                              self.df[target, PREDICTION_COLUMN_NAME, "value"])}
                for target, (fp_ratio, fn_ratio) in self.get_ratios().items()}

    @lru_cache(maxsize=None)
    def get_ratios(self):
        return {target: (cm[0,1] / cm[0,0], cm[1,0] / cm[0,0]) for target, cm in self.get_confusion_matrix().items()}

    @lru_cache(maxsize=None)
    def plot_classification(self, figsize=(16, 9)) -> Dict:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from pandas.plotting import register_matplotlib_converters

        probability_cutoff = self.probability_cutoff
        plots = {}

        for target in self.df.columns.levels[0]:
            df = self.df[target]

            # get rid of deprecation warning
            register_matplotlib_converters()

            # define grid
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])

            # plot probability
            bar = sns.lineplot(x=range(len(df)), y=df[PREDICTION_COLUMN_NAME, f"value{PROBABILITY_POSTFIX}"], ax=ax0)
            ax0.hlines(probability_cutoff, 0, len(df), color=sns.xkcd_rgb['silver'])

            # plot loss
            color = pd.Series(0, index=df.index)
            color.loc[df[(PREDICTION_COLUMN_NAME, "value")] & df[(LABEL_COLUMN_NAME, "value")]] = 1
            color.loc[(df[(PREDICTION_COLUMN_NAME, "value")] == False) & df[(LABEL_COLUMN_NAME, "value")]] = 2

            colors = {0: sns.xkcd_rgb['white'], 1: sns.xkcd_rgb['pale green'], 2: sns.xkcd_rgb['cerise']}
            palette = [colors[color_index] for color_index in np.sort(color.unique())]

            scatt = sns.scatterplot(ax=ax1,
                                    x=range(len(df)),
                                    y=df[(LOSS_COLUMN_NAME, "value")],
                                    size=df[(LOSS_COLUMN_NAME, "value")] * -1,
                                    hue=color,
                                    palette=palette)

            plt.close()
            plots[target] = fig

        return plots

    @staticmethod
    def _calculate_confusions(df):
        tp = df[df[PREDICTION_COLUMN_NAME, "value"] & df[LABEL_COLUMN_NAME, "value"]]
        fp = df[df[PREDICTION_COLUMN_NAME, "value"] & (df[LABEL_COLUMN_NAME, "value"] == False)]
        tn = df[(df[PREDICTION_COLUMN_NAME, "value"] == False) & (df[LABEL_COLUMN_NAME, "value"] == False)]
        fn = df[(df[PREDICTION_COLUMN_NAME, "value"] == False) & (df[LABEL_COLUMN_NAME, "value"])]

        return [[tp, fp],
                [fn, tn]]


class StilNeedsToBeDone():
    def _repr_html_(self):
        return self._html_()._repr_html_()

    def _html_(self, width: str = '100%'):
        # only import it needed
        from vdom.helpers import div, p, img, table, tr, td, tbody, thead, th
        import matplotlib.pyplot as plt
        import base64
        import io

        image = None
        if self.loss is not None:
            with io.BytesIO() as f:
                fig = self.plot_backtest()
                fig.savefig(f, format="png", bbox_inches='tight')
                image = base64.encodebytes(f.getvalue()).decode("utf-8")
                plt.close(fig)

        cmc = self.confusion_count()
        cml = self.confusion_loss() if self.loss is not None else np.array([[0, 0], [0, 0]])

        return div(
            table(
                thead(
                    tr(
                        th("Classification Count", style={'text-align': 'left'}),
                        th("Classification Loss", style={'text-align': 'right'})
                    )
                ),
                tbody(
                    tr(
                        td(self._matrix_table(cmc)), td(self._matrix_table(cml), style={'float': 'right'})
                    ),
                    tr(
                        td(
                            img(src=f'data:image/png;base64,{image}', style={'width': '100%'}) if image is not None else "",
                            colspan='2'
                        )
                    ),
                    tr(
                        td("FN/TP Ratio (should be < 0.5, ideally 0)"), td(f'{self.measurements["FN Ratio"]: .2f}')
                    ),
                    tr(
                        td("FP/TP Ratio (should be < 0.5, ideally 0)"), td(f'{self.measurements["FP Ratio"]: .2f}')
                    ),
                    tr(
                        td("F1 Score (should be > 0.5, ideally 1)"), td(f'{self.measurements["F1 Score"]: .2f}')
                    )
                ),
                style={'width': '100%'}
            ), style={'width': width}
        )

    def _matrix_table(self, mx: np.array):
        from vdom.helpers import table, tr, td, tbody, thead
        row_label = [[td("True")], [td("False")]]
        colors = [['green', 'orange'],
                  ['red', 'grey']]

        return table(
            thead(
                tr(
                    td("Prediction / Truth"),
                    td("True"),
                    td("False")
                )
            ),
            tbody(
                [tr(
                    row_label[row] + [td(
                        f'{mx[row, col]: .2f}', style={'color': colors[row][col]})
                        for col in range(mx.shape[1])])
                    for row in range(mx.shape[0])]
            )
        )

    def __len__(self):
        return len(self.y_true)

    def __str__(self) -> str:
        return f'\n{len(self.confusion_matrix[0,0])}\t{len(self.confusion_matrix[0,1])}' \
               f'\n{len(self.confusion_matrix[1,0])}\t{len(self.confusion_matrix[1,1])}'



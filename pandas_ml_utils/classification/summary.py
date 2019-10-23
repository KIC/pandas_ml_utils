import logging
import sys
from typing import Tuple, Union

import numpy as np
import pandas as pd

from ..model.summary import Summary
from sklearn.metrics import f1_score

log = logging.getLogger(__name__)


class ClassificationSummary(Summary):

    def __init__(self,
                 y_true: np.ndarray,
                 y_prediction: np.ndarray,
                 index: np.ndarray,
                 loss: pd.Series = None,
                 probability_cutoff: float = 0.5):
        self.y_true = y_true
        self.y_prediction = y_prediction.ravel() if len(y_prediction.shape) > 1 else y_prediction
        self.index = index
        self.loss = loss
        self.probability_cutoff = probability_cutoff
        self.confusion_matrix = self._confusion_matrix_indices()

        # immediately log some fit quality measures
        ratios = self.get_ratios()
        self.measurements = {"FN Ratio": ratios[0],
                             "FP Ratio": ratios[1],
                             "F1 Score": f1_score(y_true, y_prediction > probability_cutoff)}

        log.info(f"statistics: {self.measurements}")

    def set_probability_cutoff(self, probability_cutoff: float = 0.5):
        self.probability_cutoff = probability_cutoff
        self.confusion_matrix = self._confusion_matrix_indices()

    def _confusion_matrix_indices(self):
        index = self.index
        truth = self.y_true
        pred = self.y_prediction
        co = self.probability_cutoff

        try:
            confusion = np.array([[index[(truth == True) & (pred > co)], index[(truth == False) & (pred > co)]],
                                  [index[(truth == True) & (pred <= co)], index[(truth == False) & (pred <= co)]]])

            if len(confusion[0, 0]) <= 0:
                log.warning("Very bad fit with 0 TP")

            return confusion
        except:
            print(f"shapes: y_true: {self.y_true.shape}, y_pred: {self.y_prediction.shape}, index: {self.index.shape}")
            print("Unexpected error:", sys.exc_info()[0])
            return None

    def get_ratios(self):
        cm = self.confusion_count()
        return cm[1,0] / cm[0,0], cm[0,1] / cm[0,0]

    def plot_backtest(self,
                      y: pd.Series = None,
                      size: Union[int, pd.Series] = None,
                      figsize: Tuple[int, int] = (16, 6)):
        # only import if required
        import seaborn as sns
        import matplotlib.pyplot as plt
        from pandas.plotting import register_matplotlib_converters

        # get rid of deprecation warning
        register_matplotlib_converters()

        # check value for back test
        if self.loss is None and y is None:
            raise ValueError("No loss column defined, whether in FeaturesAndLabels nor in plot_backtest")

        # scatter plot where confusion squares are the colors, the loss is the size
        y = y if y is not None \
                else self.loss.loc[self.index] if isinstance(self.loss, pd.Series) \
                    else self.loss[self.loss.columns[0]].loc[self.index]

        # make sure we only plot actual losses
        y = y.copy().clip(upper=0)

        color = pd.Series(0, index=y.index)
        color.loc[self.confusion_matrix[0, 0]] = 1
        color.loc[self.confusion_matrix[1, 0]] = 2

        colors = {0: sns.xkcd_rgb['white'], 1: sns.xkcd_rgb['pale green'], 2: sns.xkcd_rgb['cerise']}
        palette = [colors[color_index] for color_index in np.sort(color.unique())]

        # get colors from: https://xkcd.com/color/rgb/
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_ylim([y.min() * 1.1, 1])

        scatt = sns.scatterplot(x=y.index,
                                y=y,
                                ax=ax,
                                size=size if size is not None else y * -1,
                                hue=color,
                                palette=palette)

        bar = sns.lineplot(x=y.index, y=self.y_prediction, ax=ax)
        plt.hlines(self.probability_cutoff, y.index.min(), y.index.max(), color=sns.xkcd_rgb['silver'])

        plt.close()
        return fig

    def confusion_loss(self):
        cm = self.confusion_matrix
        df = self.loss
        return np.array([[df.loc[cm[0, 0]].sum(), df.loc[cm[0, 1]].sum()],
                         [df.loc[cm[1, 0]].sum(), df.loc[cm[1, 1]].sum()]])

    def confusion_count(self):
        return np.array([
            [len(self.confusion_matrix[0, 0]), len(self.confusion_matrix[0, 1])],
            [len(self.confusion_matrix[1, 0]), len(self.confusion_matrix[1, 1])],
        ])

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



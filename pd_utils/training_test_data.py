import pandas as pd
import numpy as np
from typing import List, Tuple, Callable, Iterable, Dict, Union


class Model(object):
    pass


class FeaturesAndLabels(object):

    def __init__(self,
                 features: List[str],
                 labels: List[str],
                 target_columns: List[str] = None,
                 loss_column: str = None,
                 feature_lags: Iterable[int] = None,
                 lag_smoothing: Dict[int, Callable[[pd.Series], pd.Series]] = None,
                 probability_cutoff: float = 0.5):
        self.features = features
        self.labels = labels
        self.target_columns = target_columns
        self.loss_column = loss_column
        self.feature_lags = feature_lags
        self.lag_smoothing = lag_smoothing
        self.probability_cutoff = probability_cutoff
        self.expanded_feature_length = len(features) * sum(1 for _ in feature_lags) if feature_lags is not None else len(features)

    def len_features(self):
        return len(self.features), self.expanded_feature_length

    def len_labels(self):
        return len(self.labels)

    def __repr__(self):
        return f'FeaturesAndLabels({self.features},{self.labels},{self.target_columns},{self.loss_column},' \
               f'{self.feature_lags},{self.lag_smoothing},{self.probability_cutoff}) #{len(self.features)} ' \
               f'features expand to {self.expanded_feature_length}'

    def __str__(self):
        return self.__repr__()


class ClassificationSummary(object):

    def __init__(self,
                 y_true: np.ndarray,
                 y_prediction: np.ndarray,
                 index: np.ndarray,
                 loss: pd.Series,
                 probability_cutoff: float = 0.5):
        self.y_true = y_true
        self.y_prediction = y_prediction
        self.index = index
        self.loss = loss
        self.probability_cutoff = probability_cutoff
        self.confusion_matrix = self._confusion_matrix_indices()

    def _confusion_matrix_indices(self):
        index = self.index
        truth = self.y_true
        pred = self.y_prediction
        co = self.probability_cutoff

        confusion = np.array([[index[(truth == True) & (pred > co)], index[(truth == False) & (pred > co)]],
                              [index[(truth == True) & (pred <= co)], index[(truth == False) & (pred <= co)]]])

        return confusion

    def plot_backtest(self, y: pd.Series = None, size: Union[int, pd.Series] = None):
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
                else self.loss if isinstance(self.loss, pd.Series) \
                    else self.loss[self.loss.columns[0]]

        color = pd.Series(0, index=y.index)
        color.loc[self.confusion_matrix[0, 0]] = 1
        color.loc[self.confusion_matrix[1, 0]] = 2
        plt.figure(figsize=(16, 6))

        # get colors from: https://xkcd.com/color/rgb/
        return sns.scatterplot(x=y.index,
                               y=y,
                               size=size if size is not None else y * -1,
                               hue=color,
                               palette=[sns.xkcd_rgb['white'], sns.xkcd_rgb['pale green'], sns.xkcd_rgb['cerise']])

    def confusion_loss(self):
        cm = self.confusion_matrix
        df = self.loss
        return np.array([[df.loc[cm[0, 0]].sum()[0], df.loc[cm[0, 1]].sum()[0]],
                         [df.loc[cm[1, 0]].sum()[0], df.loc[cm[1, 1]].sum()[0]]])

    def confusion_count(self):
        return np.array([
            [len(self.confusion_matrix[0, 0]), len(self.confusion_matrix[0, 1])],
            [len(self.confusion_matrix[1, 0]), len(self.confusion_matrix[1, 1])],
        ])

    def _repr_html_(self):
        # only import it needed
        from vdom.helpers import div, h1, p, img, b, table, tr, td, tbody, thead
        import matplotlib.pyplot as plt
        import io

        # TODO also output the plot as svg
        # with io.BytesIO() as f:
        #   self.plot_backtest()
        #   plt.savefig(f, format="svg")
        #   svg = f.getvalue()
        # alternatively use base64 and <img src="data:image/png;base64,..."
        # import base64
        # base64.b16decode()

        #return display(table(
        #    # thead(tr(td(""))),
        #    tbody(
        #        [tr([td(col)] for col in row) for row in self.confusion_matrix]
        #    )
        #)

        cmc = self.confusion_count()
        cml = self.confusion_loss()


        return div(
            p("Classification Count"),
            self._matrix_table(cmc),
            p("Classification Loss"),
            self._matrix_table(cml)
        )._repr_html_()

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

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'\n{len(self.confusion_matrix[0,0])}\t{len(self.confusion_matrix[0,1])}' \
               f'\n{len(self.confusion_matrix[1,0])}\t{len(self.confusion_matrix[1,1])}'

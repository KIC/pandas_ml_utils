import logging
import numpy as np
import pandas as pd
from typing import List, Iterable
# from .features_and_Labels import FeaturesAndLabels

log = logging.getLogger(__name__)


def filtration(df: pd.DataFrame,
               label_column: str = None,
               label_correlation_threshold: float = 0.5,
               correlation_threshold: float = 0.5,
               minimum_features: int = 1,
               lags: Iterable[int] = range(100),
               show_plots: bool = True,
               figsize=(12, 10)):
    correlation_mx = df.corr()

    if show_plots:
        __plot_heatmap(correlation_mx, figsize)

    features = correlation_mx

    # find the features most correlated to the label (if provided)
    if label_column is not None:
        # drop label column
        features = features.drop(label_column, axis=0)
        features = features.drop(label_column, axis=1)

        target_vec = abs(correlation_mx[label_column]).drop(label_column)
        top_features = target_vec.sort_values(ascending=True).index.tolist()
        while target_vec[top_features[0]] < label_correlation_threshold and len(top_features) > minimum_features:
            top_features.pop(0)

        features = features.drop(top_features, axis=0)
        features = features.drop(top_features, axis=1)

        print(f"\nfeatures with correlation > {label_correlation_threshold} to {label_column}")
        print(correlation_mx[features.columns][:1])

    # then eliminate features with high correlation to each other
    while len(features) > minimum_features and __max_correlation(features) > correlation_threshold:
        index = np.unravel_index(__argmax_correlation(features), features.values.shape)
        features = features.drop(features.index[index[0]], axis=0)
        features = features.drop(features.columns[index[0]], axis=1)

    print(f"\nfiltered features with correlation < {correlation_threshold}")
    print(correlation_mx[features.columns][:1])

    if show_plots:
        __plot_heatmap(features, figsize)

    # make AR analysis of remaining features
    coefficients = []
    for feature in features.columns:
        if show_plots:
            __plot_acf(df[feature], lags, (figsize[0], int(figsize[1] / 3)))

        dff = df[[feature]]
        for lag in lags:
            dff = dff.join(df[feature].shift(lag), rsuffix=f'_{lag}')

        dff = dff.corr().dropna()
        acorr = dff.iloc[0]
        coefficients.append(acorr.values)
        print(acorr.sort_values(ascending=False)[:10])

    # find clusters of auto correlation coefficients over all features
    best_lags = np.array(coefficients).sum(axis=0)
    best_lags = (-best_lags).argsort()[1:]
    # FIXME apply threshold
    print(f"best lags are {best_lags}")


def __plot_heatmap(correlation_mx, figsize):
    try:
        # only import if needed and only plot if libraries found
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig = plt.figure(figsize=figsize)

        sns.heatmap(correlation_mx, annot=True, cmap=plt.cm.Reds)
        plt.show()
    except:
        return None


def __plot_acf(series, lags, figsize):
    try:
        from matplotlib import pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf

        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(series.name, fontsize=14)

        plot_acf(series, lags=lags, ax=ax)
        plt.show()
    except:
        return None


def __max_correlation(df):
    return (df.corr().values - np.identity(len(df.columns))).max()


def __argmax_correlation(df):
    return (df.corr().values - np.identity(len(df.columns))).argmax()
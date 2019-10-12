import logging
import pandas as pd
from typing import List

# from .features_and_Labels import FeaturesAndLabels

log = logging.getLogger(__name__)


def filtration(df: pd.DataFrame, label_columns: List[str], correlation_threshold: float = 0.5, figsize=(12, 10)):
    correlation_mx = df.corr()
    log.info(correlation_mx)

    try:
        # only import if needed and only plot if libraries found
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig = plt.figure(figsize=figsize)

        sns.heatmap(correlation_mx, annot=True, cmap=plt.cm.Reds)
    except:
        fig = None

    # select features most correlated with the target values
    for label in label_columns:
        target_vec = abs(correlation_mx[label])
        features = target_vec[target_vec > correlation_threshold].drop(label_columns)

        print(f"\nfeatures with correlation > {correlation_threshold} to {label}")
        print(features)

    # then eliminate features with high correlation
    df[features.index].corr()
    # TODO eliminate features with high correlation

    return fig

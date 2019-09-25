import pandas as pd


def mse(y: pd.DataFrame, y_hat: pd.DataFrame):
    return ((y_hat.values - y.values) ** 2).mean(axis=1)

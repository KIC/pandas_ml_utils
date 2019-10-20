import pandas as pd


def plot_correlation_matrix(df: pd.DataFrame, figsize=(12, 10)):
    correlation_matrix = df.corr()
    sorted_correlation_matrix = __sort_correlation(correlation_matrix)
    __plot_heatmap(sorted_correlation_matrix, figsize)


def __sort_correlation(correlation_matrix):
    cor = correlation_matrix.abs()
    top_col = cor[cor.columns[0]][1:]
    top_col = top_col.sort_values(ascending=False)
    ordered_columns = [cor.columns[0]] + top_col.index.tolist()
    return correlation_matrix[ordered_columns].reindex(ordered_columns)


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

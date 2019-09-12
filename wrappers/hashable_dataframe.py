import pandas as pd


class HashableDataFrame(object):

    def __init__(self, df: pd.DataFrame) -> None:
        self.df: pd.DataFrame = df

    def __getitem__(self, item: str):
        return self.df.__getitem__(item)

    def __getattr__(self, item):
        return self.df.__getattr__(item)

    def __hash__(self):
        return hash(str(self.describe()))

    def __eq__(self, other):
        try:
            pd.testing.assert_frame_equal(self.df, other.df)
            return True
        except:
            return False

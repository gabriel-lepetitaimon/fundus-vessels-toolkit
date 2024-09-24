import pandas as pd


class DFSetterAccessor:
    def __init__(self, df: pd.DataFrame, row_idx):
        self._df = df
        self._row_idx = row_idx

    def __getitem__(self, key):
        return self._df.loc[self._row_idx, key]

    def __setitem__(self, key, value):
        self._df.loc[self._row_idx, key] = value

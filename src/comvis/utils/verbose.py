import pandas as pd
import polars as pl

from comvis.utils.types import DataFrame

__all__ = ['printdf']


def printdf(df: DataFrame,
            nrows: int | None = None,
            ncols: int | None = None) -> str:
    """print polars/pandas dataframe with given row numbers"""

    if isinstance(df, pl.DataFrame):
        with pl.Config() as cfg:
            rows = df.shape[0] if nrows is None else nrows
            cols = df.shape[1] if ncols is None else ncols
            cfg.set_tbl_rows(rows)
            cfg.set_tbl_cols(cols)

            print(df)

            return df.__repr__()

    elif isinstance(df, pd.DataFrame):
        ret = df.to_markdown()
        print(ret)
        return ret

    else:
        raise TypeError('')

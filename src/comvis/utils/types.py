from io import BufferedIOBase, BufferedReader
from pathlib import Path
from typing import BinaryIO

import pandas as pd
import polars as pl

__all__ = ['PathLike', 'Series', 'DataFrame']

PathLike = str | Path | bytes | BinaryIO | BufferedIOBase | BufferedReader

Series = pd.Series | pl.Series
DataFrame = pd.DataFrame | pl.DataFrame

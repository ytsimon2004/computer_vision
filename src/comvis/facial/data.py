from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pandas as pd
import polars as pl
from typing_extensions import Self

from comvis.facial.util import DEFAULT_CACHE_DIRECTORY
from comvis.utils.types import DataFrame

__all__ = ['FacialDataSet']


class FacialDataSet:
    """For loading the dataset"""

    TRAIN_SET_FILENAME: ClassVar = 'train_set.csv'
    TEST_SET_FILENAME: ClassVar = 'test_set.csv'

    def __init__(self, train: DataFrame, test: DataFrame):
        self.train = train
        self.test = test

    @property
    def dtype(self) -> str:
        return type(self.train)

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def test_size(self) -> int:
        return len(self.test)

    @classmethod
    def load(cls, to_pandas: bool = False) -> Self:
        train = pl.read_csv((DEFAULT_CACHE_DIRECTORY / cls.TRAIN_SET_FILENAME)).drop('')
        test = pl.read_csv(DEFAULT_CACHE_DIRECTORY / cls.TEST_SET_FILENAME).drop('')
        train, test = cls._store_image(train, test, to_pandas)

        return FacialDataSet(train, test)

    @classmethod
    def from_kaggle(cls) -> Self:
        kaggle_root = Path('/kaggle/input/kul-h02a5a-computer-vision-ga1-2022')
        train = pl.read_csv(kaggle_root / cls.TRAIN_SET_FILENAME)
        test = pl.read_csv(kaggle_root / cls.TEST_SET_FILENAME)
        train, test = cls._store_image(train, test, to_pandas=True)

        return FacialDataSet(train, test)

    @classmethod
    def _store_image(cls, train: pl.DataFrame,
                     test: pl.DataFrame,
                     to_pandas: bool = False) -> tuple[DataFrame, DataFrame]:
        img_train = [
            cv2.cvtColor(np.load(f'{DEFAULT_CACHE_DIRECTORY}/train/train_{index}.npy', allow_pickle=False),
                         cv2.COLOR_BGR2RGB)
            for index, row in enumerate(train.iter_rows())
        ]
        train = (
            train.with_columns(img=pl.Series(values=img_train, dtype=pl.Object))
            .with_columns(pl.Series(np.arange(train.shape[0])).alias('id'))
        )

        #
        img_test = [
            cv2.cvtColor(np.load(f'{DEFAULT_CACHE_DIRECTORY}/test/test_{index}.npy', allow_pickle=False),
                         cv2.COLOR_BGR2RGB)
            for index, row in enumerate(test.iter_rows())
        ]

        test = (test.with_columns(img=pl.Series(values=img_test, dtype=pl.Object))
                .with_columns(pl.Series(np.arange(test.shape[0])).alias('id')))

        if to_pandas:
            return train.to_pandas(), test.to_pandas()
        else:
            return train, test

    def train_distribution(self) -> DataFrame:
        if isinstance(self.train, pl.DataFrame):
            return (
                self.train.group_by('name')
                .agg([pl.col('img').count(),
                      pl.col('class').max()])
            )
        elif isinstance(self.train, pd.DataFrame):
            return (self.train.groupby('name')
                    .agg({'img': 'count', 'class': 'max'}))

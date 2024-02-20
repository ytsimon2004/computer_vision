from io import BufferedIOBase, BufferedReader
from pathlib import Path
from typing import BinaryIO

__all__ = ['PathLike']

PathLike = str | Path | bytes | BinaryIO | BufferedIOBase | BufferedReader

from pathlib import Path

__all__ = ['DEFAULT_CACHE_DIRECTORY']

DEFAULT_CACHE_DIRECTORY = Path.home() / '.cache' / 'comvis' / 'facial'

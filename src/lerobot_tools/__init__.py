"""Utilities for converting LeRobot datasets and building their LMDB caches."""

from .dataset import LmdbFrameDataset

__all__ = ["LmdbFrameDataset", "convert", "lerobot_lmdb_build"]
__version__ = "0.1.0"


def lerobot_lmdb_build(*args, **kwargs):
    """Lazily import and run :func:`lerobot_tools.build.lerobot_lmdb_build`."""
    from .build import lerobot_lmdb_build as _lerobot_lmdb_build

    return _lerobot_lmdb_build(*args, **kwargs)


def convert(*args, **kwargs):
    """Lazily import and run :func:`lerobot_tools.conversion.convert`."""
    from .conversion import convert as _convert

    return _convert(*args, **kwargs)

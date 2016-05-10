# -*- coding: utf-8 -*-
# File: format.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from ..utils import logger
from .base import DataFlow

import random
from six.moves import range

try:
    import h5py
except ImportError:
    logger.warn("Error in 'import h5py'. HDF5Data won't be available.")
    __all__ = []
else:
    __all__ = ['HDF5Data']


"""
Adapters for different data format.
"""

class HDF5Data(DataFlow):
    """
    Zip data from different paths in an HDF5 file. Will load all data into memory.
    """
    def __init__(self, filename, data_paths, shuffle=True):
        """
        :param filename: h5 data file.
        :param data_paths: list of h5 paths to zipped. For example ['images', 'labels']
        :param shuffle: shuffle the order of all data.
        """
        self.f = h5py.File(filename, 'r')
        logger.info("Loading {} to memory...".format(filename))
        self.dps = [self.f[k].value for k in data_paths]
        lens = [len(k) for k in self.dps]
        assert all([k==lens[0] for k in lens])
        self._size = lens[0]
        self.shuffle = shuffle

    def size(self):
        return self._size

    def get_data(self):
        idxs = list(range(self._size))
        if self.shuffle:
            random.shuffle(idxs)
        for k in idxs:
            yield [dp[k] for dp in self.dps]


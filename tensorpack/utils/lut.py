#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: lut.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import six

__all__ = ['LookUpTable']

class LookUpTable(object):
    def __init__(self, objlist):
        self.idx2obj = dict(enumerate(objlist))
        self.obj2idx = {v : k for k, v in six.iteritems(self.idx2obj)}

    def size(self):
        return len(self.idx2obj)

    def get_obj(self, idx):
        return self.idx2obj[idx]

    def get_idx(self, obj):
        return self.obj2idx[obj]

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: concurrency.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import multiprocessing
import tensorflow as tf
from ..utils.concurrency import DIE
from ..tfutils.modelutils import describe_model
from ..utils import logger
from ..tfutils import *

from .common import *

__all__ = ['ParallelPredictWorker', 'QueuePredictWorker']

class ParallelPredictWorker(multiprocessing.Process):
    def __init__(self, idx, gpuid, config):
        """
        :param idx: index of the worker. the 0th worker will print log.
        :param gpuid: absolute id of the GPU to be used. set to -1 to use CPU.
        :param config: a `PredictConfig`
        """
        super(ParallelPredictWorker, self).__init__()
        self.idx = idx
        self.gpuid = gpuid
        self.config = config

    def _init_runtime(self):
        if self.gpuid >= 0:
            logger.info("Worker {} uses GPU {}".format(self.idx, self.gpuid))
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpuid)
        else:
            logger.info("Worker {} uses CPU".format(self.idx))
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        G = tf.Graph()     # build a graph for each process, because they don't need to share anything
        with G.as_default():
            if self.idx != 0:
                from tensorpack.models._common import disable_layer_logging
                disable_layer_logging()
            self.func = get_predict_func(self.config)
            if self.idx == 0:
                describe_model()

class QueuePredictWorker(ParallelPredictWorker):
    """ A worker process to run predictor on one GPU """
    def __init__(self, idx, gpuid, inqueue, outqueue, config):
        """
        :param idx: index of the worker. the 0th worker will print log.
        :param gpuid: id of the GPU to be used. set to -1 to use CPU.
        :param inqueue: input queue to get data point
        :param outqueue: output queue put result
        :param config: a `PredictConfig`
        """
        super(QueuePredictWorker, self).__init__(idx, gpuid, config)
        self.inqueue = inqueue
        self.outqueue = outqueue

    def run(self):
        self._init_runtime()
        while True:
            tid, dp = self.inqueue.get()
            if tid == DIE:
                self.outqueue.put((DIE, None))
                return
            else:
                res = PredictResult(dp, self.func(dp))
                self.outqueue.put((tid, res))

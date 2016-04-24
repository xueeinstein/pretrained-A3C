# -*- coding: UTF-8 -*-
# File: trainer.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import threading
import copy
import re
from six.moves import zip

from .base import Trainer
from ..dataflow.common import RepeatedData
from ..utils import *
from ..tfutils.summary import summary_moving_average
from ..tfutils import *

__all__ = ['SimpleTrainer', 'QueueInputTrainer', 'start_train']

class SimpleTrainer(Trainer):
    def run_step(self):
        data = next(self.data_producer)
        feed = dict(zip(self.input_vars, data))
        self.sess.run([self.train_op], feed_dict=feed)    # faster since train_op return None

    def train(self):
        model = self.model
        input_vars = model.get_input_vars()
        self.input_vars = input_vars
        cost_var = model.get_cost(input_vars, is_training=True)
        avg_maintain_op = summary_moving_average(cost_var)

        grads = self.config.optimizer.compute_gradients(cost_var)
        grads = self.process_grads(grads)

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            avg_maintain_op)

        self.init_session_and_coord()
        # create an infinte data producer
        self.data_producer = RepeatedData(self.config.dataset, -1).get_data()
        self.main_loop()

    def _trigger_epoch(self):
        if self.summary_op is not None:
            data = next(self.data_producer)
            feed = dict(zip(self.input_vars, data))
            summary_str = self.summary_op.eval(feed_dict=feed)
            self._process_summary(summary_str)

class EnqueueThread(threading.Thread):
    def __init__(self, trainer, queue, enqueue_op, raw_input_var):
        super(EnqueueThread, self).__init__()
        self.sess = trainer.sess
        self.coord = trainer.coord
        self.dataflow = RepeatedData(trainer.config.dataset, -1)

        self.input_vars = raw_input_var
        self.op = enqueue_op
        self.queue = queue
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        self.daemon = True

    def run(self):
        try:
            while True:
                for dp in self.dataflow.get_data():
                    if self.coord.should_stop():
                        return
                    feed = dict(zip(self.input_vars, dp))
                    self.op.run(feed_dict=feed, session=self.sess)
        except tf.errors.CancelledError as e:
            pass
        except Exception:
            logger.exception("Exception in EnqueueThread:")
            self.sess.run(self.close_op)
            self.coord.request_stop()
        finally:
            logger.info("Enqueue Thread Exited.")

class QueueInputTrainer(Trainer):
    """
    Trainer which builds a FIFO queue for input.
    Support multi GPU.
    """

    def __init__(self, config, input_queue=None):
        """
        :param config: a `TrainConfig` instance
        :param input_queue: a `tf.QueueBase` instance to be used to buffer datapoints.
            Defaults to a FIFO queue of size 100.
        """
        super(QueueInputTrainer, self).__init__(config)
        self.input_vars = self.model.get_input_vars()
        if input_queue is None:
            self.input_queue = tf.FIFOQueue(
                    100, [x.dtype for x in self.input_vars], name='input_queue')
        else:
            self.input_queue = input_queue

    @staticmethod
    def _average_grads(tower_grads):
        ret = []
        for grad_and_vars in zip(*tower_grads):
            v = grad_and_vars[0][1]
            try:
                grad = tf.add_n([x[0] for x in grad_and_vars]) / float(len(tower_grads))
            except AssertionError:
                logger.error("Error while processing gradients of {}".format(v.name))
                raise
            ret.append((grad, v))
        return ret

    def _get_model_inputs(self):
        """ Dequeue a datapoint from input_queue and return"""
        ret = self.input_queue.dequeue()
        if isinstance(ret, tf.Tensor): # only one input
            ret = [ret]
        assert len(ret) == len(self.input_vars)
        for qv, v in zip(ret, self.input_vars):
            qv.set_shape(v.get_shape())
        return ret

    def _single_tower_grad_cost(self):
        """ Get grad and cost for single-tower case"""
        model_inputs = self._get_model_inputs()
        cost_var = self.model.get_cost(model_inputs, is_training=True)
        grads = self.config.optimizer.compute_gradients(cost_var)
        return (grads, cost_var)

    def _multi_tower_grad_cost(self):
        logger.info("Training a model of {} tower".format(self.config.nr_tower))

        # to avoid repeated summary from each device
        collect_dedup = [tf.GraphKeys.SUMMARIES, MOVING_SUMMARY_VARS_KEY]
        kept_summaries = {}

        grad_list = []
        for i in range(self.config.nr_tower):
            with tf.device('/gpu:{}'.format(i)), \
                    tf.name_scope('tower{}'.format(i)) as scope:
                model_inputs = self._get_model_inputs()    # each tower dequeue from input queue
                cost_var = self.model.get_cost(model_inputs, is_training=True) # build tower

                # gate_gradienst=0 seems to be faster?
                grad_list.append(
                    self.config.optimizer.compute_gradients(cost_var, gate_gradients=0))

                if i == 0:
                    cost_var_t0 = cost_var
                    tf.get_variable_scope().reuse_variables()
                    for k in collect_dedup:
                        kept_summaries[k] = copy.copy(tf.get_collection(k))
                logger.info("Graph built for tower {}.".format(i))
        for k in collect_dedup:
            del tf.get_collection_ref(k)[:]
            tf.get_collection_ref(k).extend(kept_summaries[k])
        grads = QueueInputTrainer._average_grads(grad_list)
        return (grads, cost_var_t0)

    def train(self):
        enqueue_op = self.input_queue.enqueue(self.input_vars)

        grads, cost_var = self._single_tower_grad_cost() \
                if self.config.nr_tower == 0 else self._multi_tower_grad_cost()
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, cost_var)
        avg_maintain_op = summary_moving_average()

        grads = self.process_grads(grads)

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            avg_maintain_op)

        self.init_session_and_coord()
        # create a thread that keeps filling the queue
        self.input_th = EnqueueThread(self, self.input_queue, enqueue_op, self.input_vars)
        self.main_loop()

    def _start_all_threads(self):
        super(QueueInputTrainer, self)._start_all_threads()
        self.input_th.start()

    def run_step(self):
        self.sess.run([self.train_op])    # faster since train_op return None

    def _trigger_epoch(self):
        # note that summary_op will take a data from the queue
        if self.summary_op is not None:
            summary_str = self.summary_op.eval()
            self._process_summary(summary_str)

def start_train(config):
    tr = QueueInputTrainer(config)
    tr.train()

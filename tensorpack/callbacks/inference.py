# -*- coding: UTF-8 -*-
# File: inference.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
from collections import namedtuple
import six
from six.moves import zip, map

from ..dataflow import DataFlow
from ..utils import get_tqdm_kwargs, logger
from ..utils.stat import RatioCounter, BinaryStatistics
from ..tfutils import get_op_tensor_name
from .base import Callback

__all__ = ['InferenceRunner', 'ClassificationError',
        'ScalarStats', 'Inferencer', 'BinaryClassificationStats']

class Inferencer(object):
    __metaclass__ = ABCMeta

    def before_inference(self):
        """
        Called before a new round of inference starts.
        """
        self._before_inference()

    def _before_inference(self):
        pass

    def datapoint(self, dp, output):
        """
        Called after complete running every data point
        """
        self._datapoint(dp, output)

    @abstractmethod
    def _datapoint(self, dp, output):
        pass

    def after_inference(self):
        """
        Called after a round of inference ends.
        Returns a dict of statistics.
        """
        return self._after_inference()

    def _after_inference(self):
        pass

    def get_output_tensors(self):
        """
        Return a list of tensor names needed for this inference
        """
        return self._get_output_tensors()

    @abstractmethod
    def _get_output_tensors(self):
        pass

class InferenceRunner(Callback):
    """
    A callback that runs different kinds of inferencer.
    """

    IOTensor = namedtuple('IOTensor', ['index', 'isOutput'])

    def __init__(self, ds, infs, input_tensors=None):
        """
        :param ds: inference dataset. a `DataFlow` instance.
        :param infs: a list of `Inferencer` instance.
        :param input_tensor_names: list of tensors to feed the dataflow to.
            default to all the input placeholders.
        """
        assert isinstance(ds, DataFlow), type(ds)
        self.ds = ds
        if not isinstance(infs, list):
            self.infs = [infs]
        else:
            self.infs = infs
        for v in self.infs:
            assert isinstance(v, Inferencer), str(v)
        self.input_tensors = input_tensors

    def _setup_graph(self):
        self._find_input_tensors() # these are all tensor names
        self._find_output_tensors() # may be either tensor name or op name
        self.pred_func = self.trainer.get_predict_func(
                self.input_tensors, self.output_tensors)

    def _find_input_tensors(self):
        if self.input_tensors is None:
            input_vars = self.trainer.model.reuse_input_vars()
            self.input_tensors = [x.name for x in input_vars]

    def _find_output_tensors(self):
        IOTensor = InferenceRunner.IOTensor
        self.output_tensors = []
        def find_oid(t):
            tensorname = get_op_tensor_name(t)[1]
            if tensorname in self.input_tensors:
                # this inferencer needs the input dp
                return IOTensor(self.input_tensors.index(tensorname), False)
            if t in self.output_tensors:
                return IOTensor(self.output_tensors.index(t), True)
            else:
                self.output_tensors.append(t)
                return IOTensor(len(self.output_tensors) - 1, True)
        self.inf_to_tensors = [
                [find_oid(t) for t in inf.get_output_tensors()]
                for inf in self.infs]
        # list of list of (var_name: IOTensor)

    def _trigger_epoch(self):
        for inf in self.infs:
            inf.before_inference()

        sess = tf.get_default_session()
        self.ds.reset_state()
        with tqdm(total=self.ds.size(), **get_tqdm_kwargs()) as pbar:
            for dp in self.ds.get_data():
                outputs = self.pred_func(dp)
                for inf, tensormap in zip(self.infs, self.inf_to_tensors):
                    inf_output = [(outputs if k.isOutput else dp)[k.index]
                            for k in tensormap]
                    inf.datapoint(dp, inf_output)
                pbar.update()

        for inf in self.infs:
            ret = inf.after_inference()
            for k, v in six.iteritems(ret):
                try:
                    v = float(v)
                except:
                    logger.warn("{} returns a non-scalar statistics!".format(type(inf).__name__))
                    continue
                self.trainer.write_scalar_summary(k, v)

class ScalarStats(Inferencer):
    """
    Write some scalar tensor to both stat and summary.
    The output of the given Ops must be a scalar.
    The value will be averaged over all data points in the inference dataflow.
    """
    def __init__(self, names_to_print, prefix='validation'):
        """
        :param names_to_print: list of names of tensors, or just a name
        :param prefix: an optional prefix for logging
        """
        if not isinstance(names_to_print, list):
            self.names = [names_to_print]
        else:
            self.names = names_to_print
        self.prefix = prefix

    def _get_output_tensors(self):
        return self.names

    def _before_inference(self):
        self.stats = []

    def _datapoint(self, dp, output):
        self.stats.append(output)

    def _after_inference(self):
        self.stats = np.mean(self.stats, axis=0)
        assert len(self.stats) == len(self.names)

        ret = {}
        for stat, name in zip(self.stats, self.names):
            opname, _ = get_op_var_name(name)
            name = '{}_{}'.format(self.prefix, opname) if self.prefix else opname
            ret[name] = stat
        return ret

class ClassificationError(Inferencer):
    """
    Compute classification error in batch mode, from a `wrong` variable

    The `wrong` variable is supposed to be an integer equal to the number of failed samples in this batch.
    You can use `tf.nn.in_top_k` to record top-k error as well.

    This callback produce the "true" error,
    taking account of the fact that batches might not have the same size in
    testing (because the size of test set might not be a multiple of batch size).
    Therefore the result is different from averaging the error rate of each batch.
    """
    def __init__(self, wrong_var_name='wrong:0', summary_name='validation_error'):
        """
        :param wrong_var_name: name of the `wrong` variable
        :param summary_name: the name for logging
        """
        self.wrong_var_name = wrong_var_name
        self.summary_name = summary_name

    def _get_output_tensors(self):
        return [self.wrong_var_name]

    def _before_inference(self):
        self.err_stat = RatioCounter()

    def _datapoint(self, dp, outputs):
        batch_size = dp[0].shape[0]   # assume batched input
        wrong = int(outputs[0])
        self.err_stat.feed(wrong, batch_size)

    def _after_inference(self):
        return {self.summary_name: self.err_stat.ratio}

class BinaryClassificationStats(Inferencer):
    """ Compute precision/recall in binary classification, given the
    prediction vector and the label vector.
    """

    def __init__(self, pred_var_name, label_var_name, summary_prefix='val'):
        """
        :param pred_var_name: name of the 0/1 prediction tensor.
        :param label_var_name: name of the 0/1 label tensor.
        """
        self.pred_var_name = pred_var_name
        self.label_var_name = label_var_name
        self.prefix = summary_prefix

    def _get_output_tensors(self):
        return [self.pred_var_name, self.label_var_name]

    def _before_inference(self):
        self.stat = BinaryStatistics()

    def _datapoint(self, dp, outputs):
        pred, label = outputs
        self.stat.feed(pred, label)

    def _after_inference(self):
        return {self.prefix + '_precision': self.stat.precision,
                self.prefix + '_recall': self.stat.recall}

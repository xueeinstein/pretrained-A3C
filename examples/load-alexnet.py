#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: load-alexnet.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import numpy as np
import os
import argparse

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow.dataset import ILSVRCMeta

"""
Usage:
    python -m tensorpack.utils.loadcaffe PATH/TO/CAFFE/{deploy.prototxt,bvlc_alexnet.caffemodel} alexnet.npy
    ./load-alexnet.py --load alexnet.npy --input cat.png
"""

class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, (None, 227, 227, 3), 'input') ]

    def _build_graph(self, inputs):
        # img: 227x227x3
        image = inputs[0]

        with argscope([Conv2D, FullyConnected], nl=tf.nn.relu):
            l = Conv2D('conv1', image, out_channel=96, kernel_shape=11, stride=4, padding='VALID')
            l = tf.nn.lrn(l, 2, bias=1.0, alpha=2e-5, beta=0.75, name='norm1')
            l = MaxPooling('pool1', l, 3, stride=2, padding='VALID')

            l = Conv2D('conv2', l, out_channel=256, kernel_shape=5, split=2)
            l = tf.nn.lrn(l, 2, bias=1.0, alpha=2e-5, beta=0.75, name='norm2')
            l = MaxPooling('pool2', l, 3, stride=2, padding='VALID')

            l = Conv2D('conv3', l, out_channel=384, kernel_shape=3)
            l = Conv2D('conv4', l, out_channel=384, kernel_shape=3, split=2)
            l = Conv2D('conv5', l, out_channel=256, kernel_shape=3, split=2)
            l = MaxPooling('pool3', l, 3, stride=2, padding='VALID')

            # This is just a script to load model, so we ignore the dropout layer
            l = FullyConnected('fc6', l, 4096)
            l = FullyConnected('fc7', l, out_dim=4096)
        # fc will have activation summary by default. disable this for the output layer
        logits = FullyConnected('fc8', l, out_dim=1000, nl=tf.identity)
        prob = tf.nn.softmax(logits, name='output')

def run_test(path, input):
    param_dict = np.load(path).item()

    pred_config = PredictConfig(
        model=Model(),
        input_var_names=['input'],
        session_init=ParamRestore(param_dict),
        session_config=get_default_sess_config(0.9),
        output_var_names=['output']   # the variable 'output' is the probability distribution
    )
    predict_func = get_predict_func(pred_config)

    import cv2
    im = cv2.imread(input)
    assert im is not None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (227, 227))
    im = np.reshape(im, (1, 227, 227, 3)).astype('float32')
    im = im - 110
    outputs = predict_func([im])[0]
    prob = outputs[0]
    ret = prob.argsort()[-10:][::-1]
    print ret

    meta = ILSVRCMeta().get_synset_words_1000()
    print [meta[k] for k in ret]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load',
                        help='.npy model file generated by tensorpack.utils.loadcaffe',
                        required=True)
    parser.add_argument('--input', help='an input image', required=True)
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # run alexnet with given model (in npy format)
    run_test(args.load, args.input)

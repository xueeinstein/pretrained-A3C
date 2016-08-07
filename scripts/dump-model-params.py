#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: dump-model-params.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import argparse
import tensorflow as tf
import imp

from tensorpack import *
from tensorpack.tfutils import sessinit, varmanip

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config file')
parser.add_argument('--meta', help='metagraph file')
parser.add_argument(dest='model')
parser.add_argument(dest='output')
args = parser.parse_args()

assert args.config or args.meta, "Either config or metagraph must be present!"

with tf.Graph().as_default() as G:
    if args.config:
        MODEL = imp.load_source('config_script', args.config).Model
        M = MODEL()
        M.build_graph(M.get_input_vars(), is_training=False)
    else:
        M = ModelFromMetaGraph(args.meta)

    # loading...
    init = sessinit.SaverRestore(args.model)
    sess = tf.Session()
    init.init(sess)

    # dump ...
    with sess.as_default():
        if args.output.endswith('npy'):
            varmanip.dump_session_params(args.output)
        else:
            var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            var.extend(tf.get_collection(EXTRA_SAVE_VARS_KEY))
            logger.info("Variables to dump:")
            logger.info(", ".join([v.name for v in var]))
            saver = tf.train.Saver(var_list=var)
            saver.save(sess, args.output, write_meta_graph=False)

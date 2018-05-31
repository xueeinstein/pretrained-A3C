#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Bill Xue <xueeinstein@gmail.com>
import os
import cv2
import random
import argparse
import numpy as np
import tensorflow as tf
import skimage.transform as ski_transform

from tensorpack import *
from tensorpack.RL import *
from tensorpack.tfutils import get_tensors_by_names

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

NUM_ACTIONS = None
ENV_NAME = None
t_episode = 0


def normalize_activation(act, ubound=1.0, eps=1e-7):
    act_min = np.min(act)
    act_max = np.max(act)

    return (act - act_min) * ubound / (act_max - act_min + eps)


def get_player(dumpdir=None, get_origin_ob_shape=False):
    pl = GymEnv(ENV_NAME, dumpdir=dumpdir, auto_restart=False)
    if get_origin_ob_shape:
        ob_shape = pl._ob.shape

    pl = MapPlayerState(pl, lambda img: cv2.resize(img, IMAGE_SIZE[::-1]))

    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()

    pl = HistoryFramePlayer(pl, FRAME_HISTORY)
    if get_origin_ob_shape:
        return pl, ob_shape
    else:
        return pl


class MovieWriter(object):
    def __init__(self, file_name, frame_size, fps):
        """
        frame_size is (w, h)
        """
        self._frame_size = frame_size
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.vout = cv2.VideoWriter()
        success = self.vout.open(file_name, fourcc, fps, frame_size, True)
        if not success:
            print("Create movie failed: {0}".format(file_name))

    def add_frame(self, frame):
        """
        frame shape is (h, w, 3), dtype is np.uint8
        """
        self.vout.write(frame)

    def close(self):
        self.vout.release()
        self.vout = None


def play_one_episode(player, func, video_writer, verbose=False,
                     fusion_alpha=0.50):
    def f(s):
        spc = player.get_action_space()
        act, vam = func([[s]])
        act = act[0][0].argmax()
        vam = vam[0]
        ob = cv2.resize(np.uint8(s[:, :, -3:][..., ::-1]),
                        (vam.shape[1], vam.shape[0]))
        ob_fusion = fusion_alpha * vam + (1 - fusion_alpha) * ob
        ob_fusion = np.uint8(ob_fusion)
        # cv2.imshow("fusion", ob_fusion)
        # cv2.waitKey(0)
        global t_episode
        if t_episode % 4 == 0:
            video_writer.add_frame(np.concatenate((ob, ob_fusion), axis=1))

        t_episode += 1
        if random.random() < 0.001:
            act = spc.sample()
        if verbose:
            print(act)
        return act
    return np.mean(player.play_one_episode(f))


class OfflinePredictor_WithViz(OfflinePredictor):
    def __init__(self, cfg, ob_shape, feature_map_name='feature_map',
                 value_tensor_name='pred_value', input_size=84, sigma=0,
                 vis_method='vam'):
        """vis_method can be 'vam' or 'pg'."""
        super(OfflinePredictor_WithViz, self).__init__(cfg)

        self.ob_shape = ob_shape
        self.input_size = input_size
        self.sigma = sigma
        self.vis_method = vis_method
        with self.graph.as_default():
            self.feature_map_tensor, self.value_tensor = get_tensors_by_names(
                [feature_map_name, value_tensor_name])

    def _do_call(self, dp):
        """Override from OfflinePredictor <- OnlinePredictor"""
        assert len(dp) == len(self.input_tensors), \
            "{} != {}".format(len(dp), len(self.input_tensors))
        feed = dict(zip(self.input_tensors, dp))
        if self.sess is None:
            sess = tf.get_default_session()
        else:
            sess = self.sess

        if type(self.output_tensors) == list:
            outputs = sess.run([self.value_tensor, self.feature_map_tensor] +
                               self.output_tensors, feed_dict=feed)
        else:
            outputs = sess.run([self.value_tensor, self.feature_map_tensor,
                                self.output_tensors], feed_dict=feed)

        value = outputs[0]
        feature_map = outputs[1]
        outputs_res = outputs[2:]

        if self.vis_method == 'vam':
            vis_maps = self.get_value_attention_maps(sess, value, feature_map)
        elif self.vis_method == 'pg':
            vis_maps = self.get_pixel_gradients_saliency_map(sess, feed)
        else:
            raise Exception('Unsupported visualization method: {}'.format(
                self.vis_method))

        return outputs_res, vis_maps

    def get_value_attention_maps(self, sess, value, feature_map):
        value_attention_maps = []
        for v, fm in zip(value, feature_map):
            # get contribution of every hidden unit
            fm_silence_collect = []
            for i in range(feature_map.shape[-1]):
                fm_silence = np.transpose(fm, [2, 0, 1]).copy()
                fm_silence[i] = np.zeros(fm_silence[i].shape)
                fm_silence_collect.append(fm_silence)

            feed = {self.feature_map_tensor:
                    np.transpose(np.array(fm_silence_collect), [0, 2, 3, 1])}
            v_ = sess.run(self.value_tensor, feed_dict=feed)
            contrib = np.abs((v_ - v) / v)

            # get value attention map
            contrib = np.expand_dims(np.expand_dims(contrib, 0), 0)
            contrib = np.tile(contrib, [fm.shape[0], fm.shape[1], 1])

            weighted_fm = np.multiply(fm, contrib)
            weighted_fm = np.sum(weighted_fm, axis=2)
            weighted_fm = normalize_activation(weighted_fm)

            upscale = self.input_size / fm.shape[0]
            vam = ski_transform.pyramid_expand(weighted_fm, upscale=upscale,
                                               sigma=self.sigma)
            vam = np.asarray(vam * 255, dtype=np.uint8)
            vam = cv2.applyColorMap(vam, cv2.COLORMAP_JET)
            vam = cv2.resize(vam, (self.ob_shape[1], self.ob_shape[0]))
            value_attention_maps.append(vam)

        return np.array(value_attention_maps)

    def get_pixel_gradients_saliency_map(self, sess, feed):
        # get image input tensor
        image_input_tensor = None
        for t in self.input_tensors:
            if 'state' in str(t.name):
                image_input_tensor = t
                break

        pixel_grads_tensor = tf.gradients(self.value_tensor, image_input_tensor)
        pixel_grads = sess.run(pixel_grads_tensor, feed_dict=feed)
        saliency_map = pixel_grads[0][..., -3:]
        saliency_map = normalize_activation(saliency_map)
        saliency_map = np.asarray(saliency_map * 255, dtype=np.uint8)

        vis_maps = []
        for sm in saliency_map:
            sm = cv2.applyColorMap(sm, cv2.COLORMAP_JET)
            sm = cv2.resize(sm, (self.ob_shape[1], self.ob_shape[0]))
            vis_maps.append(sm)

        return np.array(vis_maps)


class Model(ModelDesc):
    def _get_inputs(self):
        assert NUM_ACTIONS is not None
        return [InputDesc(tf.float32, (None,) + IMAGE_SHAPE3, 'state'),
                InputDesc(tf.int32, (None,), 'action'),
                InputDesc(tf.float32, (None,), 'futurereward')]

    def _get_NN_prediction(self, image):
        image = image / 255.0
        with argscope(Conv2D, nl=tf.nn.relu):
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)

        # [None, 10, 10, 64]
        self.feature_map = tf.identity(l, name='feature_map')

        l = FullyConnected('fc0', self.feature_map, 512, nl=tf.identity)
        l = PReLU('prelu', l)
        policy = FullyConnected('fc-pi', l, out_dim=NUM_ACTIONS, nl=tf.identity)
        value = FullyConnected('fc-v', l, 1, nl=tf.identity)
        return policy, value

    def _build_graph(self, inputs):
        state, action, futurereward = inputs
        policy, self.value = self._get_NN_prediction(state)
        self.value = tf.squeeze(self.value, [1], name='pred_value')
        self.logits = tf.nn.softmax(policy, name='logits')


def run_submission(cfg, output, nr, vis_method):
    player, ob_shape = get_player(dumpdir=output, get_origin_ob_shape=True)
    predfunc = OfflinePredictor_WithViz(cfg, ob_shape, vis_method=vis_method)
    logger.info("Start evaluation: ")
    global t_episode
    for k in range(nr):
        if k != 0:
            t_episode = 0
            player.restart_episode()
        vis_video = os.path.join(output, "ep_{}.mp4".format(k))
        video_writer = MovieWriter(vis_video, (ob_shape[1] * 2, ob_shape[0]), 8)
        score = play_one_episode(player, predfunc, video_writer)
        video_writer.close()
        print("Score:", score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', help='environment name')
    parser.add_argument('-g', '--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='pretrained model directory', default='pretrained')
    parser.add_argument('-e', '--episode', help='number of episodes to run',
                        type=int, default=20)
    parser.add_argument('--output', help='output directory', default='~/Video')
    parser.add_argument('-v', '--vis-method', help='visualization method: vam, pg',
                        default='vam')
    args = parser.parse_args()

    env=args.env
    ENV_NAME = '{}-v0'.format(env)
    args.load = os.path.expanduser('{}/{}.tfmodel'.format(args.load, ENV_NAME))
    args.output = os.path.expanduser(os.path.join(args.output, env))

    assert ENV_NAME
    logger.info("Environment Name: {}".format(ENV_NAME))
    p = get_player()
    del p    # set NUM_ACTIONS

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cfg = PredictConfig(
        model=Model(),
        session_init=SaverRestore(args.load),
        input_names=['state'],
        output_names=['logits'])
    run_submission(cfg, args.output, args.episode, args.vis_method)

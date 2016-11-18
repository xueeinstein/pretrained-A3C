#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz.py
# Credit: zxytim

import numpy as np
import os, sys
import io
import cv2
from .fs import mkdir_p
from .argtools import shape2d

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

__all__ = ['pyplot2img', 'build_patch_list', 'pyplot_viz',
        'dump_dataflow_images']

def pyplot2img(plt):
    buf = io.BytesIO()
    plt.axis('off')
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    rawbuf = np.frombuffer(buf.getvalue(), dtype='uint8')
    im = cv2.imdecode(rawbuf, cv2.IMREAD_COLOR)
    buf.close()
    return im

def pyplot_viz(img, shape=None):
    """ use pyplot to visualize the image
        Note: this is quite slow. and the returned image will have a border
    """
    plt.clf()
    plt.axes([0,0,1,1])
    plt.imshow(img)
    ret = pyplot2img(plt)
    if shape is not None:
        ret = cv2.resize(ret, shape)
    return ret

def minnone(x, y):
    if x is None: x = y
    elif y is None: y = x
    return min(x, y)

def build_patch_list(patch_list,
        nr_row=None, nr_col=None, border=None,
        max_width=1000, max_height=1000,
        shuffle=False, bgcolor=255):
    """
    This is a generator.
    patch_list: bhw or bhwc
    :param border: defaults to 0.1 * max(image_width, image_height)
    """
    patch_list = np.asarray(patch_list)
    if patch_list.ndim == 3:
        patch_list = patch_list[:,:,:,np.newaxis]
    assert patch_list.ndim == 4 and patch_list.shape[3] in [1, 3], patch_list.shape
    if shuffle:
        np.random.shuffle(patch_list)
    ph, pw = patch_list.shape[1:3]
    if border is None:
        border = int(0.1 * max(ph, pw))
    mh, mw = max(max_height, ph + border), max(max_width, pw + border)
    if nr_row is None:
        nr_row = minnone(nr_row, max_height / (ph + border))
    if nr_col is None:
        nr_col = minnone(nr_col, max_width / (pw + border))

    canvas = np.zeros((nr_row * (ph + border) - border,
             nr_col * (pw + border) - border,
             patch_list.shape[3]), dtype='uint8')

    def draw_patch(plist):
        cur_row, cur_col = 0, 0
        canvas.fill(bgcolor)
        for patch in plist:
            r0 = cur_row * (ph + border)
            c0 = cur_col * (pw + border)
            canvas[r0:r0+ph, c0:c0+pw] = patch
            cur_col += 1
            if cur_col == nr_col:
                cur_col = 0
                cur_row += 1

    nr_patch = nr_row * nr_col
    start = 0
    while True:
        end = start + nr_patch
        cur_list = patch_list[start:end]
        if not len(cur_list):
            return
        draw_patch(cur_list)
        yield canvas
        start = end

def dump_dataflow_images(df, index=0, batched=True,
        number=300, output_dir=None,
        scale=1, resize=None, viz=None, flipRGB=False, exit_after=True):
    if output_dir:
        mkdir_p(output_dir)
    if viz is not None:
        viz = shape2d(viz)
        vizsize = viz[0] * viz[1]
    if resize is not None:
        resize = tuple(shape2d(resize))
    vizlist = []

    df.reset_state()
    cnt = 0
    while True:
        for dp in df.get_data():
            if not batched:
                imgbatch = [dp[index]]
            else:
                imgbatch = dp[index]
            for img in imgbatch:
                cnt += 1
                if cnt == number:
                    if exit_after:
                        sys.exit()
                    else:
                        return
                if scale != 1:
                    img = img * scale
                if resize is not None:
                    img = cv2.resize(img, resize)
                if flipRGB:
                    img = img[:,:,::-1]
                if output_dir:
                    fname = os.path.join(output_dir, '{:03d}.jpg'.format(cnt))
                    cv2.imwrite(fname, img)
                if viz is not None:
                    vizlist.append(img)
            if viz is not None and len(vizlist) >= vizsize:
                patch = next(build_patch_list(
                    vizlist[:vizsize],
                    nr_row=viz[0], nr_col=viz[1]))
                cv2.imshow("df-viz", patch)
                cv2.waitKey()
                vizlist = vizlist[vizsize:]


if __name__ == '__main__':
    import cv2
    imglist = []
    for i in range(100):
        fname = "{:03d}.png".format(i)
        imglist.append(cv2.imread(fname))
    for idx, patch in enumerate(build_patch_list(
            imglist, max_width=500, max_height=200)):
        of = "patch{:02d}.png".format(idx)
        cv2.imwrite(of, patch)

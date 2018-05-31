#!/usr/bin/env python
import subprocess as pc

EPISODES = 1
NUM_GPUS = 8
VIS='vam'
LOAD_DIR = 'pretrained'
VIDEO_DIR = '~/Video'

games = list(map(str.strip, open('game_list.txt').readlines()))

games = games[:23]
print('Games:')
for g in games:
    print('\t', g)

gpu = 0
for game in games:
    cmd = ('examples/A3C-Gym/visualize-atari.py {} -g {} -e {} --load {} --output {} -v {}'
           .format(game, gpu, EPISODES, LOAD_DIR, VIDEO_DIR, VIS))
    print('GPU', gpu, '>>>', cmd)
    pc.Popen(cmd, shell=True)
    gpu = (gpu + 1) % NUM_GPUS

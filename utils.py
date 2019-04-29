
import os
import logging
import shutil
from datetime import datetime

import math
import numpy as np
import numba
from PIL import Image, ImageColor
import copy
import random

randomobj = random.Random(0)
PALETTE = []
PALETTE.extend([0,0,0])
colormap = copy.copy(ImageColor.colormap)
del colormap['black']
del colormap['white']
colors = list(colormap)
randomobj.shuffle(colors)
for i in range(1,255):
    PALETTE.extend(ImageColor.getrgb(colors[i % len(colors)]))
PALETTE.extend([255,255,255])

def visualize_colored(colored, classes, prefix='', visualize_dir='visulize_results'):
    os.makedirs(visualize_dir, exist_ok=True)
    batch_size = colored.shape[0]
    for idx in range(batch_size):
        cimg = Image.fromarray(np.uint8(colored[idx]), mode='L').convert('P')
        cimg.putpalette(PALETTE)
        cimg = cimg.convert('RGB')
        cimg.save(os.path.join(visualize_dir, 'union{}{}.jpg'.format(prefix, idx)))
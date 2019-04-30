import os
import logging
import shutil
from datetime import datetime
import torchvision
import math
import numpy as np
import numba
from PIL import Image, ImageColor
import copy
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
topil = torchvision.transforms.ToPILImage()

def pca_image(vec_graph):
    shape = vec_graph.shape
    assert len(shape)==3
    size = shape[1:]
    
    # normalize all vectors so their 2-norms are 1
    norms = np.linalg.norm(vec_graph, axis=0)
    vec_graph /= norms

    vecs = vec_graph.reshape(shape[0], -1).transpose()
    
    results = PCA(2).fit_transform(vecs).reshape((size[0], size[1], 2))

    minval = results.min()
    maxval = results.max()
    results -= minval
    results /= maxval-minval
    results = (results *255).astype(np.uint8)

    return results

def visualize_colored(input,colored, classes, prefix='', visualize_dir='visulize_results'):
    os.makedirs(visualize_dir, exist_ok=True)
    batch_size = colored.shape[0]
    for idx in range(batch_size):
        iimg = topil(input[idx])
        cimg = Image.fromarray(np.uint8(colored[idx]), mode='L').convert('P')
        cimg.putpalette(PALETTE)
        cimg = cimg.convert('RGB')
        blendc = Image.blend(cimg, iimg, 0.5)
        blendc.save(os.path.join(visualize_dir, 'union{}{}.jpg'.format(prefix, idx)))

def analyse(classPred):
    batchSize,h,w = classPred.shape
    for batch in range(batchSize):
        totolClass = np.max(classPred[batch])
        for i in range(totolClass):
            cnt = (classPred[batch] == i).sum()
            plt.scatter([i],[cnt],s=3)
        plt.show()
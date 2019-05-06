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

def calCnt(colormap,totalClass):
    cnt = np.zeros((totalClass)).astype(np.int64)
    for i in range(totalClass):
        cnt[i] = (colormap == i).sum()
    return cnt

def eatBlk(colorMap,clsOfColor,totalClass):
    #step 1: break blocks to small connected blocks
    #step 2: count big background or lane neighbors of each block
    #step 3: if a block only have one neighbor, it is surrounded by the big neighbor, so the big one "eat" it  
    h,w = colorMap.shape
    xDelta = [0,1,0,-1]
    yDelta = [1,0,-1,0]
    visited = np.zeros((h,w)).astype(np.int8)
    cnt = np.zeros((totalClass)).astype(np.int64)
    for i in range(totalClass):
        cnt[i] = (colorMap == i).sum()

    for x in range(h):
        for y in range(w):
            if (visited[x,y] != 0):
                continue
            if ((clsOfColor[colorMap[x,y]] == 1) or colorMap[x,y] == 0): #only break blocks that belong to background or lane
                #do color fill to find connected block
                queue = [(x,y)]
                head = 0
                tail = 0
                rootColor = colorMap[x,y]
                clsOfColor[totalClass] = clsOfColor[rootColor]
                visited[x,y] = 1
                currentCnt = 0
                while (head <= tail):
                    xNow,yNow = queue[head]
                    head += 1
                    currentCnt += 1
                    colorMap[xNow,yNow] = totalClass
                    for i in range(4):
                        xNew = xNow + xDelta[i]
                        yNew = yNow + yDelta[i]
                        if (xNew < 0 or xNew >= h or yNew < 0 or yNew >= w):
                            continue
                        if (visited[xNew,yNew] == 0 and colorMap[xNew,yNew] == rootColor):
                            queue.append((xNew,yNew))
                            visited[xNew,yNew] = 1
                            tail += 1
                if (rootColor == 0 and currentCnt > 20000):
                    colorMap[colorMap == totalClass] = 0
                else:
                    totalClass += 1
    cnt = np.zeros((totalClass)).astype(np.int64)
    for i in range(totalClass):
        cnt[i] = (colorMap == i).sum()
    #find neighbors for each block
    neighbor = np.zeros((totalClass,totalClass)).astype(np.int8)
    for x in range(h):
        for y in range(w):
            for i in range(4):
                xNew = x + xDelta[i]
                yNew = y + yDelta[i]
                if (xNew < 0 or xNew >= h or yNew < 0 or yNew >= w):
                    continue
                if clsOfColor[colorMap[xNew,yNew]] != 2 and cnt[colorMap[xNew,yNew]] > 30000:
                    neighbor[colorMap[x,y],colorMap[xNew,yNew]] = 1
    #eat the small block
    for i in range(1,totalClass):
        if (neighbor[i].sum() == 1):
            father = np.argwhere(neighbor[i] == 1)[0,0]
            if ((clsOfColor[i] == 2 and cnt[i] < 800) or (clsOfColor[i] != 2 and cnt[i] < cnt[father])):
                colorMap[colorMap == i] = father
    print(totalClass)
    return totalClass


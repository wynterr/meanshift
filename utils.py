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
from operator import itemgetter

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

def compose_img(a, *args):
    imgs = [a]
    imgs.extend(args)
    img = Image.new('RGB', (a.size[0]*len(imgs) + 3*(len(imgs)-1), a.size[1]))
    for idx, i in enumerate(imgs):
        img.paste(i, box=((a.size[0]+3)*idx,0))
    return img
    
def visualize_colored(input,colored, classes, prefix='', visualize_dir='visulize_results'):
    os.makedirs(visualize_dir, exist_ok=True)
    batch_size = colored.shape[0]
    for idx in range(batch_size):
        iimg = topil(input[idx])
        cimg = Image.fromarray(np.uint8(colored[idx]), mode='L').convert('P')
        cimg.putpalette(PALETTE)
        cimg = cimg.convert('RGB')
        clsimg = Image.fromarray(np.uint8(classes[idx]), mode='L').convert('P')
        clsimg.putpalette(PALETTE)
        clsimg = clsimg.convert('RGB')
        blendcls = Image.blend(clsimg, iimg, 0.5)
        blendc = Image.blend(cimg, iimg, 0.5)
        composed = compose_img(blendc, blendcls)
        composed.save(os.path.join(visualize_dir, 'union{}{}.jpg'.format(prefix, idx)))

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

def eatBlk(colorMap,clsOfColor,totalClass,centers,featureMap,threshold):
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
                    centers[totalClass] += featureMap[:,xNow,yNow]
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
                    centers[totalClass] = np.zeros_like(featureMap[:,0,0])
                else:
                    centers[totalClass] /= currentCnt
                    centers[totalClass] /= np.linalg.norm(centers[totalClass])
                    totalClass += 1
    cnt = np.zeros((totalClass)).astype(np.int64)
    for i in range(totalClass):
        cnt[i] = (colorMap == i).sum()
    #find neighbors for each block
    #neighbor = np.zeros((totalClass,totalClass)).astype(np.int8)
    neighbor = {}
    for x in range(h):
        for y in range(w):
            for i in range(4):
                xNew = x + xDelta[i]
                yNew = y + yDelta[i]
                if (xNew < 0 or xNew >= h or yNew < 0 or yNew >= w):
                    continue
                if clsOfColor[colorMap[xNew,yNew]] != 2 and cnt[colorMap[xNew,yNew]] > 30000 and colorMap[xNew,yNew] != colorMap[x,y]:
                    neighbor[colorMap[x,y],colorMap[xNew,yNew]] = 1
    lastNeighbor = np.zeros((totalClass)).astype(np.int64)  #if cntNeight[i] == 0, this is the only neighbor of i
    cntNeighbor = np.zeros((totalClass)).astype(np.int64)  #the number of neighbors a block has
    for record in neighbor.keys():
        cntNeighbor[record[0]] += 1
        lastNeighbor[record[0]] = record[1]
    #eat the small block
    for i in range(1,totalClass):
        if (cntNeighbor[i] == 1):
            father = lastNeighbor[i]
            if ((clsOfColor[i] == 2 and cnt[i] < 800) or (clsOfColor[i] != 2 and \
                    ((clsOfColor[i] == 1 and cnt[i] < threshold) or (clsOfColor[i] == 0)) \
                        and cnt[i] < cnt[father])):
                colorMap[colorMap == i] = father
                cnt[father] +=cnt[i] 
                cnt[i] = 0
    return totalClass

def getThreshold(clsMap):
    cnt = np.bincount(clsMap.reshape(-1))
    cnt[0] = 0
    if (np.max(cnt) < 50000):
        return 5000
    elif (np.max(cnt) < 80000):
        return 10000
    elif (np.max(cnt) < 170000):
        return 25000
    else:
        return 50000

def trim_color(pred_ins, pred_cls, trim_threshold=10):
    # the function clears those colored blocks that is greatly smaller than others
    # it the block is trim_threshold times smaller than the larger block, or saying that, having a gap ther, it is trimmed.
    num_classes = pred_cls.max()
    if num_classes==0:
        return pred_ins
    pred_ins = pred_ins.astype(np.int64)
    outcomes = []
    for i in range(1, num_classes+1):

        # select the instance prediction of the class
        class_colored = pred_ins * (pred_cls==i).astype(np.int64)
        
        ins_cnt = np.bincount(class_colored.reshape(-1))
        ins_cnt = list(enumerate(ins_cnt))[1:]
        # sort by count of pixels, from larger amount to smaller
        sorted_cnt = sorted(ins_cnt, key=itemgetter(1), reverse=True)
        print(sorted_cnt)
        # trim all smaller blocks if one block is to be trimmed
        trimming = False
        for idx, (ins, cnt) in enumerate(sorted_cnt[1:]):
            if not trimming and cnt*trim_threshold < sorted_cnt[idx][1]:
                # take the idx here carefully
                trimming = True
            if trimming:
                class_colored[class_colored==ins] = 0
        outcomes.append(class_colored)
    
    # stack up all outcomes to form new instance prediction
    res = np.stack(outcomes).sum(0)
    return res

def perspectiveTrans(img,clsMap):
    pass

def curveFit(points):
    #fit a 3rd polynomial for the corresponding lane line
    #points is the list of all points belong to the line
    #first we sample some points from the list and fit a curve to the sampled points
    n = points.shape[0]
    sortedByX = sorted(points,key = itemgetter(1))
    plt.scatter(sortedByX[:,0],sortedByX[:,1],s=3)
    plt.show()

output = np.load('myoutput/output0.npy')
b,h,w = output.shape

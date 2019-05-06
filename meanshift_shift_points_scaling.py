import numpy as np
import torch
import random
from numpy import random as nr
import math
import matplotlib.pyplot as plt
import numba
from utils import *
class Meanshifter:
    def __init__(self,bandwidth,minDist2Shift,maxDist2Merge,minBlockSize):
        self.bandwidth = bandwidth      #bandwidth of the kernel
        self.minDist2Shift = minDist2Shift      #the min distance for a point shifting to a new one
        self.maxDist2Merge = maxDist2Merge      #the max distance for two class center to merge
        self.cntL = 0
        self.minBlockSize = minBlockSize     #block with size smaller than this will be merged
    
    @numba.jit
    def cosine_distance(self,x,y):
        return 1-((x * y).sum(axis = 1))/(np.linalg.norm(x)*np.linalg.norm(y,axis=1)+1e-12)
    @numba.jit
    def euclidean_distance(self,x,y):
        return np.sqrt(((x - y)**2).sum(axis=1))
    @numba.jit
    def gaussian_kernel(self,x,y):
        #cos_distance = self.cosine_distance(x,y)
        euclidean_distance = self.euclidean_distance(x,y)
        val = (1/(self.bandwidth* 0.25 *math.sqrt(2*math.pi))) * np.exp(-0.5*((euclidean_distance / (self.bandwidth * 0.25)))**2)
        return val

    @numba.jit
    def meanshift(self,inputClass,inputFeature):
        self.cntL = 0
        inputClass[np.isnan(inputClass)] = 0
        inputFeature[np.isnan(inputFeature)] = 0
        shape = inputFeature.shape
        batchSize = shape[0]
        h = shape[2]
        w = shape[3]
        output = np.zeros((batchSize,h,w)).astype(np.int64)
        for batch in range(batchSize):
            self.cntL = 0
            shifted = np.zeros((h,w,shape[1]))
            allPoints = inputFeature[batch,:,:,:].transpose().reshape(-1,shape[1]) #turn feature map to 1D
            inputFeature[batch,:,:,:] = (allPoints / (np.tile(np.linalg.norm(allPoints,axis=1),[shape[1],1]).transpose() + 1e-12))\
                                        .reshape(w,h,shape[1]).transpose()   #normalization
            allPoints = inputFeature[batch,:,:,:].transpose().reshape(-1,shape[1])  #turn feature map to 1D
            allClasses = inputClass[batch,:,:].transpose().reshape(-1)  #turn class map to 1D
            labeld = np.zeros((h,w)).astype(np.int32)     #mark if the point has been assign to a class or not
            centers = np.zeros((5000,shape[1]))   #feature center of the class
            clsOfBlk = np.zeros((5000)).astype(np.int8)  #indicate which class a color block belong to
            print('batch: %d'%batch)
            currentClass = 1     #class 0 is background
            xOrder = list(range(h // 2,-1,-1)) + list(range((h // 2) + 1,h))  #reorder x axis,just random trying
            for x in xOrder:
                for y in range(w):
                    if (labeld[x,y] != 0 or currentClass > 250):
                        continue
                    print("point: %d,%d"%(x,y))
                    if (inputClass[batch,x,y] == 0):
                        continue
                    point = inputFeature[batch,:,x,y]
                    while(True):
                        #calculate weight for each point using the gaussian kernel
                        #shift until the distance of shifting is smaller than the threshold
                        weights = np.multiply(self.gaussian_kernel(point,allPoints),(allClasses == inputClass[batch,x,y]) * ((labeld == 0).transpose().reshape(-1)))      ##only points have same class count
                        tiledWeights = np.tile(weights,[shape[1],1])
                        newPoint = np.multiply(tiledWeights.transpose(),allPoints).sum(axis = 0) / weights.sum()
                        newPoint = newPoint / np.linalg.norm(newPoint)
                        #print("distance: %f"%self.euclidean_distance(point,newPoint.reshape(1,-1)))
                        if (self.euclidean_distance(point,newPoint.reshape(1,-1)) < self.minDist2Shift):
                            break
                        else:
                            point = newPoint
                    shifted[x,y] = point
                    clsOfBlk[currentClass] = inputClass[batch,x,y]
                    self.makeCluster(inputFeature[batch,:,:,:],(inputClass[batch,:,:] == inputClass[batch,x,y]),\
                                        point,currentClass,labeld,output[batch,:,:],centers)
                    currentClass += 1
            totalClass = currentClass
            #eatBlk(output[batch],clsOfBlk,totalClass)
            ##merge blocks whose distance between centers is smaller than threshold
            merged = np.zeros((5000)).astype(np.int8)
            for color1 in range(1,totalClass):
                for color2 in range(color1 + 1,totalClass):
                    if ((merged[color2] == 0) and clsOfBlk[color1] == clsOfBlk[color2]\
                        and self.euclidean_distance(centers[color1],centers[color2].reshape(1,-1)) < self.maxDist2Merge):
                        output[batch,output[batch] == color2] = color1
                        merged[color2] = 1
            cnt = calCnt(output[batch],totalClass)
            ##merge very small blocks to the block whose cordinate center is closest to its
            for color1 in range(1,totalClass):
                if (merged[color1] or cnt[color1] > self.minBlockSize):
                    continue
                minDist = 999999
                index = 0
                for color2 in range(1,totalClass):
                    if (color1 != color2 and cnt[color2] > 3000 and clsOfBlk[color1] == clsOfBlk[color2]\
                        and self.euclidean_distance(centers[color1],centers[color1].reshape(1,-1)) < minDist):
                        minDist = self.euclidean_distance(centers[color1],centers[color1].reshape(1,-1))
                        index = color2
                output[batch,output[batch] == color1] = index
                cnt[index] += cnt[color1]
                cnt[color1] = 0
                merged[color1] = 1
            totalClass = eatBlk(output[batch],clsOfBlk,totalClass)
            totalClass = eatBlk(output[batch],clsOfBlk,totalClass)
            totalClass = eatBlk(output[batch],clsOfBlk,totalClass)
            cnt = calCnt(output[batch],totalClass)
            ##merge very small blocks to the block whose cordinate center is closest to its
            for color1 in range(1,totalClass):
                if (merged[color1] or cnt[color1] > self.minBlockSize):
                    continue
                minDist = 999999
                index = 0
                for color2 in range(1,totalClass):
                    if (color1 != color2 and cnt[color2] > 3000 and clsOfBlk[color1] == clsOfBlk[color2]\
                        and self.euclidean_distance(centers[color1],centers[color1].reshape(1,-1)) < minDist):
                        minDist = self.euclidean_distance(centers[color1],centers[color1].reshape(1,-1))
                        index = color2
                output[batch,output[batch] == color1] = index
                cnt[index] += cnt[color1]
                cnt[color1] = 0
                merged[color1] = 1
            ##compress color count
            currentClass = 1
            for color in range(1,totalClass):
                if ((output[batch] == color).sum() != 0):
                    output[batch,output[batch] == color] = currentClass
                    currentClass += 1
            print(currentClass)
        return output        

    @numba.jit
    def makeCluster(self,points,sameClass,point,classNum,labeld,output,centers):
        #gather all points that have a distance smaller than threshold with point and assign them a new color
        h,w = output.shape
        eucDist = self.euclidean_distance(point,points.transpose().reshape(-1,20)).reshape((w,h)).transpose()
        #print('done calculating')
        cnt = 0
        for i in range(h):
            for j in range(w):
                #print(cosDist[i,j])
                if (sameClass[i,j] and labeld[i,j] == 0 and eucDist[i,j] < self.bandwidth):
                    centers[classNum] += points[:,i,j]
                    labeld[i,j] = 1
                    output[i,j] = classNum
                    cnt += 1
        centers[classNum] /= cnt
        centers[classNum] /= np.linalg.norm(centers[classNum])
        self.cntL += cnt
        print(cnt)
        print("total:%d"%self.cntL)


    @numba.jit
    def group(self,points,classes,output):
        currentClass = 1     #class 0 is background
        classCenter = []     #center point of each class
        classOfCenter = []    #the class that the center belongs to
        shape = points.shape
        h = shape[0]
        w = shape[1]
        for i in range(h):
            for j in range(w):
                minDist = 2
                index = -1
                #finding the nearest center
                for k in range(currentClass - 1):
                    dist = self.euclidean_distance(points[i,j],classCenter[k].reshpe(1,-1))
                    if (dist < minDist and classOfCenter[k] == classes[i,j]):
                        minDist = dist
                        index = k
                if (index != -1 and minDist < self.maxDist2Merge):
                    output[i,j] = index + 1
                else:
                    classCenter.append(points[i,j])
                    classOfCenter.append(classes[i,j])
                    output[i,j] = currentClass
                    currentClass += 1
        
    @numba.jit
    def preProcess(self,points,classes,labeld):
        ##calculate the number of the map-related-neighbors(of the 8 neighbors) that a point has a distance smaller than the bandwidth
        ##if the number is smaller than 3, take it as a single class
        _,h,w = points.shape
        shift = [-1,0,1]
        currentClass = 1
        for x in range(h):
            print(x)
            for y in range(w):
                cnt = 0
                if (classes[x,y] == 0):
                    continue
                for xShift in shift:
                    for yShift in shift:
                        if (xShift == 0 and yShift == 0):
                            continue
                        if (x + xShift < 0 or x + xShift >= h or y + yShift < 0 or y + yShift >= w):
                            continue
                        if (classes[x,y] != classes[x + xShift,y + yShift]):
                            continue
                        dist = self.euclidean_distance(points[:,x,y],points[:,x + xShift,y + yShift].reshape(1,-1))
                        if (dist < self.bandwidth):
                            cnt += 1
                if (cnt < 2):
                    labeld[x,y] = 1
        return currentClass
    
    @numba.jit
    def showWeigtDistribution(self,featureMap,classMap):
        dim,h,w = featureMap.shape
        ws = []
        for i in range(1000):
            x = random.randint(0,h - 1)
            y = random.randint(0,w - 1)
            print(i)
            weights = np.multiply(self.gaussian_kernel(featureMap[:,x,y],featureMap.transpose().reshape(-1,dim)),(classMap.transpose().reshape(-1) == classMap[x,y]))
            ws.append(weights.sum())
        ws.sort()
        plt.scatter(np.arange(0,len(ws)),ws,s = 3)
        plt.show()

if (__name__ == '__main__'):
    inputData = np.load('ipt/input10.npy')
    classData = np.load('fms/class10.npy')
    featureData = np.load('fms/instance10.npy')
    print(featureData.shape)
    print(inputData.shape)
    print('data successfully loaded')
    output = Meanshifter(1.1,1e-3,1.1,500).meanshift(classData,featureData)
    np.save('output',output)
    print('done')
    analyse(output)
    visualize_colored(torch.from_numpy(inputData),output,classData)

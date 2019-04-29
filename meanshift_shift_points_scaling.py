import numpy as np
from numpy import random as nr
import math
import matplotlib.pyplot as plt
import numba
from utils import visualize_colored
class Meanshifter:
    def __init__(self,bandwidth,minDist2Shift,maxDist2Merge):
        self.bandwidth = bandwidth      #bandwidth of the kernel
        self.minDist2Shift = minDist2Shift      #the min distance for a point shifting to a new one
        self.maxDist2Merge = maxDist2Merge      #the max distance for two class center to merge
        self.cntL = 0
    
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
        inputClass[np.isnan(inputClass)] = 0
        inputFeature[np.isnan(inputFeature)] = 0
        shape = inputFeature.shape
        batchSize = shape[0]
        h = shape[2]
        w = shape[3]
        output = np.zeros((batchSize,h,w)).astype(np.int64)
        for batch in range(batchSize):
            shifted = np.zeros((h,w,shape[1]))
            allPoints = inputFeature[batch,:,:,:].transpose().reshape(-1,shape[1])
            inputFeature[batch,:,:,:] = (allPoints / (np.tile(np.linalg.norm(allPoints,axis=1),[shape[1],1]).transpose() + 1e-12)).reshape(w,h,shape[1]).transpose()
            allPoints = inputFeature[batch,:,:,:].transpose().reshape(-1,shape[1])
            allClasses = inputClass[batch,:,:].transpose().reshape(-1)
            labeld = np.zeros((h,w)).astype(np.int32)     #mark if the point has been assign to a class or not
            #print('batch: %d'%batch)
            currentClass = self.preProcess(inputFeature[batch,:,:,:],inputClass[batch,:,:],labeld)     #class 0 is background
            for x in range(h):
                for y in range(w):
                    if (labeld[x,y] != 0):
                        continue
                    output[batch,x,y] = currentClass
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
                    self.makeCluster(inputFeature[batch,:,:,:],(inputClass[batch,:,:] == inputClass[batch,x,y]),point,currentClass,labeld,output[batch,:,:])
                    currentClass += 1
        print(currentClass)
        return output        
    @numba.jit
    def makeCluster(self,points,sameClass,point,classNum,labeld,output):
        h,w = output.shape
        eucDist = self.euclidean_distance(point,points.transpose().reshape(-1,20)).reshape((w,h)).transpose()
        #print('done calculating')
        cnt = 0
        for i in range(h):
            for j in range(w):
                #print(cosDist[i,j])
                if (sameClass[i,j] and labeld[i,j] == 0 and eucDist[i,j] < self.bandwidth):
                    labeld[i,j] = 1
                    output[i,j] = classNum
                    cnt += 1
        self.cntL += cnt
        #print(cnt)
        #print("total:%d"%self.cntL)


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

if (__name__ == '__main__'):
    classData = np.fromfile('class.txt',dtype = np.int64).reshape(2,576,1024)
    ori = np.fromfile('feature.txt',dtype = np.float32)
    featureData = np.zeros(2 * 20 * 576 * 1024)
    featureData[:ori.shape[0]] = ori
    featureData = featureData.reshape(2,20,576,1024)
    print('data successfully loaded')
    output = Meanshifter(0.2,1e-3,0.4).meanshift(classData,featureData)
    print('done')
    #output = color_image_fill(classData,featureData)
    print(output)
    visualize_colored(output,classData)

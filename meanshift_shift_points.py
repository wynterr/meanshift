import numpy as np
from numpy import random as nr
import math
import matplotlib.pyplot as plt
import numba

class Meanshifter:
    def __init__(self,bandwidth,minDist2Shift,maxDist2Merge):
        self.bandwidth = bandwidth      #bandwidth of the kernel
        self.minDist2Shift = minDist2Shift      #the min distance for a point shifting to a new one
        self.maxDist2Merge = maxDist2Merge      #the max distance for two class center to merge
    
    @numba.jit
    def cosine_distance(self,x,y):
        return 1-((x * y).sum(axis = 1))/(np.linalg.norm(x)*np.linalg.norm(y)+1e-12)
    
    @numba.jit
    def gaussian_kernel(self,x,y):
        cos_distance = self.cosine_distance(x,y)
        val = (1/(self.bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((cos_distance / self.bandwidth))**2)
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
            allClasses = inputClass[batch,:,:].reshape(-1)
            labled = np.zeros((h,w))     #mark if the point has been assign to a class or not
            print('batch: %d'%batch)
            currentClass = 1     #class 0 is background
            for x in range(h):
                for y in range(w):
                    if (labled[x,y] != 0):
                        continue
                    labled[x,y] = 1
                    output[batch,x,y] = currentClass
                    print("point: %d,%d"%(x,y))
                    if (inputClass[batch,x,y] == 0):
                        continue
                    point = inputFeature[batch,:,x,y]
                    while(True):
                        #calculate weight for each point using the gaussian kernel
                        #shift until the distance of shifting is smaller than the threshold
                        weights = np.multiply(self.gaussian_kernel(point,allPoints),(allClasses == inputClass[batch,x,y]))      ##only points have same class count
                        tiledWeights = np.tile(weights,[shape[1],1])
                        newPoint = np.multiply(tiledWeights.transpose(),allPoints).sum(axis = 0) / weights.sum()
                        #print("distance: %f"%self.cosine_distance(point,newPoint.reshape(1,-1)))
                        if (self.cosine_distance(point,newPoint.reshape(1,-1)) < self.minDist2Shift):
                            break
                        else:
                            point = newPoint
                    shifted[x,y] = point
                    self.makeCluster(inputFeature[batch,:,:,:],(inputClass[batch,:,:] == inputClass[batch,x,y]),point,currentClass,labled,output[batch,:,:])
                    currentClass += 1
            #output[batch,:,:] = self.group(shifted,inputClass[batch,:,:],output[batch,:,:])
        return output        
    @numba.jit
    def makeCluster(self,points,sameClass,point,classNum,labled,output):
        h,w = output.shape
        cosDist = self.cosine_distance(point,points.reshape(20,-1).transpose()).reshape((h,w))
        #print('done calculating')
        for i in range(h):
            for j in range(w):
                if (sameClass[i,j] and labled[i,j] == 0 and cosDist[i,j] < self.bandwidth):
                    labled[i,j] = 1
                    output[i,j] = classNum


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
                    dist = self.cosine_distance(points[i,j],classCenter[k].reshpe(1,-1))
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
        
                            
if (__name__ == '__main__'):
    classData = np.fromfile('class.txt',dtype = np.int64).reshape(2,576,1024)
    ori = np.fromfile('feature.txt',dtype = np.float32)
    featureData = np.zeros(2 * 20 * 576 * 1024)
    featureData[:ori.shape[0]] = ori
    featureData = featureData.reshape(2,20,576,1024)
    print('data successfully loaded')
    output = Meanshifter(0.2,1e-6,0.4).meanshift(classData,featureData)
    print(output)
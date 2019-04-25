import numpy as np
from numpy import random as nr
import math
from datamaker import *
import matplotlib.pyplot as plt
import numba

def visualize(dataset,dataClass = None,centers = None):
    plt.scatter(dataset[0,0,:,:].reshape(-1),dataset[0,1,:,:].reshape(-1),c=('red' if dataClass is None else dataClass[0,:,:].reshape(-1) * 10),s = 5)
    if (centers is not None):
        plt.scatter(centers[:,0],centers[:,1])
    plt.show()

class Meanshifter:
    def __init__(self,radius,minDis4ClassCenter,minDis4Shift):
        self.radius = radius
        self.minDis4ClassCenter = minDis4ClassCenter     #the threshold of the distance between centers of different classes
        self.minDis4Shift = minDis4Shift      #the threshold of the distance between the new and old center when shifting
        self.maxClass = 200
        self.rootOfClass = np.linspace(0,self.maxClass - 1,self.maxClass).astype(np.int64)    #the root class of each class, for merging classes
        self.weightThreshold = (1/(self.radius*math.sqrt(2*math.pi))) * np.exp(-0.5*(1)**2)   #the minimum weight to be counted when counting neighbors

    def reset(self):
        self.rootOfClass = np.linspace(0,self.maxClass - 1,self.maxClass).astype(np.int64)

    @numba.jit
    def cosine_distance(self,x,y):
        return 1-((x * y).sum(axis = 1))/(np.linalg.norm(x)*np.linalg.norm(y)+1e-12)

    def gaussian_kernel(self,x,y, bandwidth):
        cos_distance = self.cosine_distance(x,y)
        val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((cos_distance / bandwidth))**2)
        return val

    def getRoot(self,x):
        if (self.rootOfClass[x] == x):
            return x
        else:
            self.rootOfClass[x] = self.getRoot(self.rootOfClass[x])
            return self.rootOfClass[x]

    def union(self,x,y):
        rootx = self.getRoot(x)
        rooty = self.getRoot(y)
        if (rootx != rooty):
            self.rootOfClass[rootx] = rooty
    

    @numba.jit
    def meanshift(self,inputClass,inputFeature):    
        #shape of inputFeature should be (batch * dimension of classes/features * h * w)
        #shape of inputClass should be (batch * h * w)
        shape = inputFeature.shape
        batchSize = shape[0]
        h = shape[2]
        w = shape[3]
        output = np.zeros((batchSize,h,w)).astype(np.int64)
        for batch in range(batchSize):
            currentClass = 1     #notice:begin from 1
            features = inputFeature[batch,:,:,:]     #only reference here
            classes = inputClass[batch,:,:]
            visit = np.zeros((h,w)).astype(np.int64)       #mark whether the point has been visited
            prob = np.zeros((h,w,self.maxClass))    #indicate the probility of a point belong to a class
            centerOfClass = np.zeros((self.maxClass,shape[1]))
            for x in range(h):
                for y in range(w):
                    if (np.isnan(features[:,x,y]).sum() > 0):
                        continue
                    print(x,y)
                    if (visit[x,y] != 0 or classes[x,y] == 0 or currentClass >= self.maxClass):
                        continue
                    center = features[:,x,y]
                    while (True):
                        neighbors = []
                        #print(center)
                        for x1 in range(h):
                            for y1 in range(w):
                                if (np.isnan(features[:,x1,y1]).sum() > 0):
                                    continue
                                #search for all points whose euclidean distance to the center is smaller than the radius
                                if (self.cosine_distance(center,features[:,x1,y1].reshape(1,-1)) < self.radius and classes[x1,y1] == classes[x,y]):
                                    neighbors.append(features[:,x1,y1])
                                    prob[x1,y1,currentClass] += 1
                                    visit[x1,y1] = 1
                        #print(len(neighbors))
                        neighbors = np.array(neighbors)
                        #calculate the new center after shifting by attaching a weight to each neighbor using a gaussian kernel
                        weights = self.gaussian_kernel(center,neighbors,self.radius)
                        #print(weights)
                        tiledWeights = np.tile(weights,[len(center),1])
  #                      print(tiledWeights)
                        newCenter = np.multiply(tiledWeights.transpose(),neighbors).sum(axis = 0) / weights.sum()
                        #print("-----------------------------------")
                        if (self.cosine_distance(center,newCenter.reshape(1,-1)) < self.minDis4Shift):
                            break
                        else:
                            center = newCenter
                    centerOfClass[currentClass] = center
                    if (currentClass < self.maxClass - 1):
                        currentClass += 1
            totalClass = currentClass
            finalClass = totalClass
            for i in range(1,totalClass):
                #select the closest center of other classes, if distance between them is smaller than the threshold, merge the two class
                minDist = 999999999
                index = 0
                for j in range(1,totalClass):
                    if (j != i and np.linalg.norm(centerOfClass[i] - centerOfClass[j]) < self.minDis4ClassCenter):
                        #index = j
                        self.union(i,j)
                #if (index > 0 and minDist < self.minDis4ClassCenter):
                    #self.union(i,index)
            #merge the probility to the root
            for i in range(1,totalClass):
                root = self.getRoot(i)
                print(i,root)
                print(centerOfClass[i])
                if (root != i):
                    finalClass -= 1
                    prob[:,:,root] += prob[:,:,i]
                    prob[:,:,i] -= prob[:,:,i]
            #choose the most possible class for each point
            for x in range(h):
                for y in range(w):
                    maxProb = 0
                    for c in range(1,totalClass):
                        if (prob[x,y,c] > maxProb):
                            maxProb = prob[x,y,c]
                            output[batch,x,y] = c
        print(totalClass)
        print(finalClass)
        #visualize(inputFeature,output,centerOfClass[1:totalClass])
        return output



if __name__ == '__main__':
    #data = generateBatch(1,100,10,1000)
    #visualize(data)
    #output = Meanshifter(120,130,1).meanshift(np.ones((1,100,10)),data)
    #visualize(data,output)
    classData = np.fromfile('class.txt',dtype = np.int64).reshape(2,576,1024)
    ori = np.fromfile('feature.txt',dtype = np.float32)
    featureData = np.zeros(2 * 20 * 576 * 1024)
    featureData[:ori.shape[0]] = ori
    featureData = featureData.reshape(2,20,576,1024)
    print('data successfully loaded')
    meanshifter = Meanshifter(1e-1,0.3,1e-8)
    output = meanshifter.meanshift(classData,featureData)
import numpy as np
from numpy import random as nr
import math
from datamaker import *
import matplotlib.pyplot as plt
import numba
class MeanShifter:
    def __init__(self,inputData,r):
        self.r = r
        self.data = inputData
        shape = inputData.shape
        self.n = shape[0]
        self.dim = shape[1]
        self.maxClass = 1000
        self.prob = np.zeros((self.n,self.maxClass)).astype(np.int64)
        self.rootOfClass = np.linspace(0,self.maxClass - 1,self.maxClass).astype(np.int64)
        self.minDistance = 1
        self.minClassDistance = 100
        self.centerOfClass = np.zeros((self.maxClass,self.dim))
        self.totalClass = 0

    def gaussian_kernel(self,distance, bandwidth):
        euclidean_distance = np.sqrt(((distance)**2).sum(axis=0))
        val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((euclidean_distance / bandwidth))**2)
        return val

    def getRoot(self,x):
        if (self.rootOfClass[x] == x):
            return x
        else:
            self.rootOfClass[x] = self.getRoot(self.rootOfClass[x])
            return self.rootOfClass[x]
    @numba.jit
    def meanshift(self):
        currentClass = 0
        visit = np.zeros(self.n)
        for i in range(self.n):
            if (visit[i] > 0):
                continue
            #print(str(i) + ":")
            
            center = self.data[i]
            while (True):
                ccnt = 0
                sphere = []
                for j in range(self.n):
                    if (np.linalg.norm(center - self.data[j]) < self.r):
                        sphere.append(self.data[j])
                        visit[j] = 1
                        ccnt += 1
                        self.prob[j,currentClass] += 1
                #print(ccnt)
                weightSum = 0
                newCenter = np.zeros(self.dim)
                for y in sphere:
                    weight = self.gaussian_kernel(center - y,self.r)
                    newCenter += weight * y
                    weightSum += weight
                newCenter /= weightSum
                #print(newCenter)
                if (np.linalg.norm(center - newCenter) < self.minDistance):
                    break
                else:
                    center = newCenter
            self.centerOfClass[currentClass] = center
            
            minDist = 999999999
            index = 0
            for j in range(currentClass):
                if (np.linalg.norm(center - self.centerOfClass[j]) < minDist):
                    minDist = np.linalg.norm(center - self.centerOfClass[j])
                    index = j
            if (minDist < self.minClassDistance):
                self.rootOfClass[currentClass] = index
            
            currentClass += 1
        totalClass = currentClass
        print(totalClass)
        finalClass = totalClass
        for i in range(totalClass):
            root = self.getRoot(i)
            if (root != i):
                self.prob[:,root] += self.prob[:,i]
                self.prob[:,i] = np.zeros_like(self.prob[:,i])
                finalClass -= 1
        print(finalClass)
        output = np.zeros(self.n)
        for i in range(self.n):
            maxProb = 0
            for j in range(totalClass):
                if (self.prob[i,j] > maxProb):
                    maxProb = self.prob[i,j]
                    output[i] = j
        self.totalClass = totalClass
        return output

def visualize(dataset,dataClass = None,centers = None):
    plt.scatter(dataset[:,0],dataset[:,1],c=('red' if dataClass is None else dataClass * 10),s = 5)
    if (centers is not None):
        plt.scatter(centers[:,0],centers[:,1])
    plt.show()


if (__name__ == '__main__'):
    inputData = generateDataMultiNormal(100,10,1000)
    visualize(inputData)
    meanshifter = MeanShifter(inputData,100)
    output = meanshifter.meanshift()
    centers = np.zeros((meanshifter.totalClass,2))
    for i in range(meanshifter.totalClass):
        centers[i] = meanshifter.centerOfClass[i]
    visualize(inputData,output,centers)


    